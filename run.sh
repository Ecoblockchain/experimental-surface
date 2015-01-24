#!/bin/bash

set -e

blender=""

if [ "$(uname)" == "Darwin" ]; then
  blender="/Applications/Blender/blender.app/Contents/MacOS/blender"
else
  blender="/usr/share/blender/blender"
fi

if [ ! -e "$blender" ]; then
  echo "Expected to find blender at: $blender"
  echo "Make sure blender is installed, or fix the path inside this script"
  exit 1
fi

base_mesh_name="$1"

here=$(pwd)
base="$here/$base_mesh_name"
generate="$here/generate.py"

res_folder="$here/res/"

if [ ! -d "$res_folder" ]; then
  mkdir $res_folder
fi

if [ $# -eq 0 ]; then
  echo "No base mesh provided. Try something like:"
  echo "./run.sh base_mesh/sphere.blend"
  exit 1
fi

if [ $# -gt 1 ]; then
  echo "too many parameters, expected 1"
  exit 1
fi

if [ ! -f "$base" ]; then
  echo "no such file: $base";
  exit 1;
fi


"$blender" "$base" -b -P "$generate"

