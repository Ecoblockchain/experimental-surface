# Experimental Surface

This is a three dimensional implementation of the same ideas (more or less)
that inspired the Differential algorithm
(https://github.com/inconvergent/differential_ani), which I wrote a while back.

The faces of the mesh are affected by their neighboring faces over
time—rejecting non-neighboring faces that are too close, and keeping a
comfortale distance to their actual neighbors. The surface is gradually
remeshed to account for it's growing surface area.

Run using:

    ./run.sh base_mesh/sphere.blend

This uses `sphere.blend` as the seed. Note that the seed object must be named
`geom`.

## Requirements

This script requires `Scipy` to be installed. (It uses `scipy.spatial.Delaunay`)

### Example Result

![res](ex/res.png?raw=true "res")

![res1](ex/res1.png?raw=true "res1")

-----------
http://inconvergent.net

