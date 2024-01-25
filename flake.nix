# flake.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    #       ↑ Swap it for your system if needed
    #       "aarch64-linux" / "x86_64-darwin" / "aarch64-darwin"
    pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true;
						     config.cudaSupport = true;};
    overlay = import ./overlay.nix;
				     
in {
	devShells.${system}.default = pkgs.mkShell { 
      packages = [ 
		pkgs.python311
		pkgs.poetry
		pkgs.magma
		overlay
		pkgs.python311Packages.pyarrow
		pkgs.python311Packages.packaging
		pkgs.python311Packages.pip
		pkgs.python311Packages.numpy
		pkgs.python311Packages.pandas
		pkgs.python311Packages.ipykernel
		pkgs.python311Packages.jupyter-core
		pkgs.python311Packages.ipywidgets
		pkgs.python311Packages.scikit-learn
		pkgs.python311Packages.notebook
		pkgs.python311Packages.torch
		pkgs.python311Packages.torchvision
		pkgs.python311Packages.torchinfo
		pkgs.python311Packages.botorch
  		pkgs.python311Packages.ax
		pkgs.python311Packages.opencv4
		pkgs.python311Packages.matplotlib
		pkgs.python311Packages.joblib
	 ];
       nativeBuildInputs = [overlay];	 

        # Workaround in linux: python downloads ELF's that can't find glibc
  # You would see errors like: error while loading shared libraries: name.so: cannot open shared object file: No such file or directory
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    #pkgs.lib
    # Add any missing library needed
    # You can use the nix-index package to locate them, e.g. nix-locate -w --top-level --at-root /lib/libudev.so.1
  ];
  
  
  # Put the venv on the repo, so direnv can access it
  POETRY_VIRTUALENVS_IN_PROJECT = "true";
  POETRY_VIRTUALENVS_PATH = "{project-dir}/.venv";
  
  # Use python from path, so you can use a different version to the one bundled with poetry
  POETRY_VIRTUALENVS_PREFER_ACTIVE_PYTHON = "true";


    };
  };
}
