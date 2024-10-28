oss = ["ubuntu-latest", "macos-12", "macos-14", "windows-latest"]
toolchains = ["stable", "beta", "nightly"]


for toolchain in toolchains:
    for os in oss:
        filename = "test_{}_{}.yml".format(toolchain, os)
        contents = f'''\
on: [push, pull_request]

name: Test-Suite {toolchain} {os}

jobs:
  CI-{toolchain}-{os}:
    uses: ./.github/workflows/reuse.yml
    with:
      toolchain: {toolchain}
      os: {os}
'''
        with open(filename, "w") as f:
            f.write(contents)
