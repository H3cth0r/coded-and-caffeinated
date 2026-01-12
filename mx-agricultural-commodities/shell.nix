{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "jupyter-pip-shell";

  buildInputs = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv

    # REQUIRED for pyzmq / numpy / scipy wheels
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi

    source .venv/bin/activate

    pip install --upgrade pip

    pip install \
      notebook \
      jupyterlab \
      jupyterlab-vim \
      numpy \
      pandas \
      matplotlib \
      plotly \
      scikit-learn \
      lxml \


    echo "Jupyter ready. Run: jupyter lab"
  '';
}
