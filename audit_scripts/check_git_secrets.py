"""
Audit Script: Check for secrets/sensitive files in git tracked files.
Run from project root: python audit_scripts/check_git_secrets.py
"""
import subprocess, os

# Get all tracked files
result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
files = result.stdout.strip().split("\n")

# Check for .env files in git
env_files = [f for f in files if f.endswith(".env") or ".env." in f]
print(f"Tracked .env files: {env_files}")
for ef in env_files:
    if os.path.exists(ef):
        with open(ef) as fp:
            content = fp.read()
        # Check for real API keys (non-placeholder)
        for line in content.splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                v = v.strip()
                placeholder_hints = ["your_", "example", "here", "xxx", "placeholder"]
                if v and not any(h in v.lower() for h in placeholder_hints):
                    print(f"  POTENTIAL REAL KEY in {ef}: {k.strip()}=<{len(v)} chars>")

print("\nTracked model/binary artifacts (>1MB):")
for f in files:
    if os.path.exists(f):
        sz = os.path.getsize(f)
        if sz > 1_000_000:
            print(f"  {sz//1024//1024:>4}MB  {f}")
