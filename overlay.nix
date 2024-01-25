# overlay.nix
final: prev:
{
  magma = prev.magma.overrideAttrs (oldAttrs: {
    cmakeFlags = oldAttrs.cmakeFlags ++ [
      "-DCMAKE_C_FLAGS=-DADD_"
      "-DCMAKE_CXX_FLAGS=-DADD_"
      "-DFORTRAN_CONVENTION:STRING=-DADD_"
    ];
  });
}
