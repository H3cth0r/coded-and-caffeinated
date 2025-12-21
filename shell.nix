{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python311;
  pythonPackages = python.pkgs;
  llvmPkgs = pkgs.llvmPackages_17;
in pkgs.mkShell {
  buildInputs = [
    # Python
    python
    pythonPackages.pip
    pythonPackages.virtualenv
    pythonPackages.numpy

    # Compilers
    pkgs.gcc
    llvmPkgs.clang
    llvmPkgs.lld
    llvmPkgs.llvm

    # System deps
    pkgs.xorg.libX11
  ];

  # This is CRITICAL for TinyGrad
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    llvmPkgs.llvm
  ];

  shellHook = ''
    # Sanity check: LLVM must be visible
    echo "Using llvm-config at: $(which llvm-config)"
    llvm-config --version

    # Create and activate virtualenv
    if [ ! -d "venv" ]; then
      virtualenv venv
    fi
    source venv/bin/activate
  '';
}
