# NeRFGL
NeRFGL is an implementation of the NeRF rendering method using OpenGL, 
using compute and fragment shaders for GPU accelerated ML.
## Capabilities
- Given a set of images of a scene with defined camera transforms, trains a model
to render that scene.
- Given a trained model, renders a camera view to an interactive window
- Given a trained model and a camera path, renders a video of the camera path

## Concerns
-Current implementation renders incredibly slowly on forward pass with only 2 layers
and any more than 30 nodes per layer, but NeRF paper has 8 layers of 256 channels.
-Not currently separating view density and position like in NeRF paper
-Hardcoded GPU backprop will probably never come close to PyTorch or TF because 
I am not a large team of machine learning experts.

