"""Freeze committed sources; simulator assets remain host-owned shared data."""
import json,subprocess,tarfile,shutil,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
source=HERE/'source'
if source.exists():raise FileExistsError(source)
source.mkdir()
versions={}
for name,repo,target,paths in [
 ('hitl',ROOT/'.codex-worktrees/fragile-tossing-integration',source,['src','scripts','configs','pyproject.toml']),
 ('kindergarden',ROOT/'.codex-worktrees/kindergarden-tossing3d-mat',source/'reference/kindergarden',['src','pyproject.toml']),
 ('kinder-baselines',ROOT/'.codex-worktrees/kinder-baselines-tossing3d-mat',source/'reference/kinder-baselines',['kinder-models/src','kinder-models/pyproject.toml'])]:
 target.mkdir(parents=True,exist_ok=True)
 sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();versions[name]=sha
 archive=HERE/f'{name}-source.tar'
 with archive.open('wb') as out:subprocess.run(['git','-C',str(repo),'archive',sha,*paths],stdout=out,check=True)
 with tarfile.open(archive) as t:t.extractall(target)
 archive.unlink()
shutil.copytree(ROOT/'artifacts/serialization-relaunch-20261008/source-original/robocode',source/'robocode',symlinks=False,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
assets=source/'reference/kindergarden/src/kinder/envs/dynamic3d/models/assets/mimiclabs_scenes'
original=ROOT/'reference/kindergarden/src/kinder/envs/dynamic3d/models/assets/mimiclabs_scenes'
for name in ['meshes','textures']:(assets/name).symlink_to(original/name,target_is_directory=True)
versions['robocode_tree_sha256']=hashlib.sha256(b''.join(p.relative_to(source/'robocode').as_posix().encode()+p.read_bytes() for p in sorted((source/'robocode').rglob('*')) if p.is_file())).hexdigest()
(HERE/'source-versions.json').write_text(json.dumps(versions,indent=2)+'\n')
print(json.dumps(versions))
