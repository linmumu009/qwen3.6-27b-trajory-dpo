"""Decrypt, verify and selectively restore the experimental evidence backup.

Requires Python 3.10+ and cryptography. No network access is performed.
"""
import argparse
from contextlib import closing
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tarfile

MAGIC = b'EXPBK001'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(4 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def decrypted_parts(folder, key, manifest):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aes = AESGCM(key)
    for i, row in enumerate(manifest['parts'], 1):
        p = folder / row['name']
        if p.stat().st_size != row['bytes'] or digest(p) != row['sha256']:
            raise ValueError(f'Encrypted part failed SHA256: {p.name}')
        blob = p.read_bytes()
        if blob[:8] != MAGIC or int.from_bytes(blob[8:12], 'big') != i:
            raise ValueError(f'Invalid part header: {p.name}')
        yield aes.decrypt(blob[12:24], blob[24:], MAGIC + blob[8:12])


class PartReader(io.RawIOBase):
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.current = memoryview(b'')

    def readable(self):
        return True

    def readinto(self, target):
        if not self.current:
            try:
                self.current = memoryview(next(self.chunks))
            except StopIteration:
                return 0
        n = min(len(target), len(self.current))
        target[:n] = self.current[:n]
        self.current = self.current[n:]
        return n


def unpack(folder, key_file, destination, manifest):
    key = bytes.fromhex(key_file.read_text(encoding='utf8').strip())
    if len(key) != 32:
        raise ValueError('The decryption key must contain 64 hexadecimal characters')
    destination.mkdir(parents=True, exist_ok=True)
    chunks = decrypted_parts(folder, key, manifest)
    reader = io.BufferedReader(PartReader(chunks))
    count = 0
    with tarfile.open(fileobj=reader, mode='r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile():
                raise ValueError(f'Unsafe archive member: {member.name}')
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, target.open('wb') as out:
                shutil.copyfileobj(src, out)
            count += 1
    # Drain remaining authenticated chunks even if tar has reached its end marker.
    while reader.read(1024 * 1024):
        pass
    print(f'Decrypted and authenticated {len(manifest["parts"])} parts; extracted {count} files')


def check_archive(folder, key_file, snapshot, manifest):
    """Check the real encrypted archive against local inputs without a second copy."""
    key = bytes.fromhex(key_file.read_text(encoding='utf8').strip())
    reader = io.BufferedReader(PartReader(decrypted_parts(folder, key, manifest)))
    count = 0
    with tarfile.open(fileobj=reader, mode='r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile():
                raise ValueError('Unsafe archive member')
            source = snapshot.joinpath(*name.parts)
            h = hashlib.sha256()
            with archive.extractfile(member) as src:
                while b := src.read(4 * 1024 * 1024):
                    h.update(b)
            if source.stat().st_size != member.size or digest(source) != h.hexdigest():
                raise ValueError(f'Archive content differs from local input: {member.name}')
            count += 1
    while reader.read(1024 * 1024):
        pass
    print(f'Verified {len(manifest["parts"])} encrypted parts and {count} archive members against local source SHA256')


def verify_objects(snapshot):
    objects = snapshot / 'local' / 'objects'
    count = 0
    for p in objects.glob('*.gz'):
        h = hashlib.sha256()
        with gzip.open(p, 'rb') as f:
            while b := f.read(4 * 1024 * 1024):
                h.update(b)
        if h.hexdigest() != p.stem:
            raise ValueError(f'Object failed SHA256: {p.name}')
        count += 1
    available = {p.stem for p in objects.glob('*.gz')}
    for database in (snapshot / 'local').glob('objects_*.sqlite3'):
        with closing(sqlite3.connect(str(database))) as db:
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError(f'Database integrity check failed: {database.name}')
            for sha, compressed in db.execute('SELECT sha256, content_gzip FROM objects'):
                h = hashlib.sha256()
                with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode='rb') as f:
                    while b := f.read(4 * 1024 * 1024):
                        h.update(b)
                if h.hexdigest() != sha:
                    raise ValueError(f'Object failed SHA256: {sha}')
                available.add(sha)
                count += 1
    for p in snapshot.glob('*_manifest.jsonl'):
        for line in p.open(encoding='utf8'):
            row = json.loads(line)
            if 'error' in row or row['sha256'] not in available:
                raise ValueError(f'Missing backed-up record: {row["path"]}')
    print(f'Verified {count} original-content SHA256 objects and all manifest references')


def materialize(snapshot, destination, match):
    count = 0
    databases = [sqlite3.connect(str(p)) for p in (snapshot / 'local').glob('objects_*.sqlite3')]
    for manifest in snapshot.glob('*_manifest.jsonl'):
        host = manifest.name.removesuffix('_manifest.jsonl')
        for line in manifest.open(encoding='utf8'):
            row = json.loads(line)
            if 'error' in row or (match and match not in row['path']):
                continue
            path = PurePosixPath(row.get('relative_path', row['path'].lstrip('/')))
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Invalid manifest path')
            target = destination / host / Path(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            compressed_path = snapshot / 'local' / 'objects' / (row['sha256'] + '.gz')
            if compressed_path.exists():
                src = gzip.open(compressed_path, 'rb')
            else:
                value = None
                for db in databases:
                    value = db.execute('SELECT content_gzip FROM objects WHERE sha256=?', (row['sha256'],)).fetchone()
                    if value:
                        break
                if value is None:
                    raise ValueError(f'Missing object: {row["sha256"]}')
                src = gzip.GzipFile(fileobj=io.BytesIO(value[0]), mode='rb')
            with src, target.open('wb') as out:
                shutil.copyfileobj(src, out)
            if 'mtime_ns' in row:
                os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))
            count += 1
    for db in databases:
        db.close()
    print(f'Restored {count} files into {destination}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts', type=Path)
    p.add_argument('--key-file', type=Path)
    p.add_argument('--snapshot', type=Path, default=Path(__file__).parent / 'restored_snapshot')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--check-archive', action='store_true', help='Verify encrypted archive against an existing local snapshot, without extracting')
    p.add_argument('--materialize', type=Path)
    p.add_argument('--match', help='Only restore original paths containing this text')
    a = p.parse_args()
    if a.parts is None:
        a.parts = Path(__file__).parent / 'encrypted'
        location = Path(__file__).parent / 'encrypted_parts_location.txt'
        if location.exists():
            candidate = Path(location.read_text(encoding='utf8').strip())
            if candidate.is_dir():
                a.parts = candidate
    if a.key_file:
        manifest = json.loads((Path(__file__).parent / 'public_manifest.json').read_text(encoding='utf8'))
        if a.check_archive:
            check_archive(a.parts, a.key_file, a.snapshot, manifest)
        else:
            unpack(a.parts, a.key_file, a.snapshot, manifest)
    if a.verify:
        verify_objects(a.snapshot)
    if a.materialize:
        materialize(a.snapshot, a.materialize, a.match)
