### Synthetic motion on synthetic images
The images are randomly generated with squares flying in the image.
The square does not have any texture, nor the background.
Image resolution: 32x32x3.

motion range = 1 corresponds to 9+1 motion classes.
motion range = 2 corresponds to 25+1 motion classes.
motion range = 3 corresponds to 49+1 motion classes.
motion range = 5 corresponds to 121+1 motion classes.

input: multiple previous frames (i.e. 28x28x3x4)
output: local motion (i.e. 28x28x9), disappear pixels (i.e. 28x28x1) and next frame (i.e. 28x28x3)

| Local motion | Training Loss (%) |
| ------------- | ----------- | ----------- |
| motion range = 1, unsupervised 3 frames, UNet | |
| motion range = 2, unsupervised 3 frames, UNet | |
| motion range = 3, unsupervised 3 frames, UNet | |
| motion range = 5, unsupervised 3 frames, UNet | |

Take Home Message:

