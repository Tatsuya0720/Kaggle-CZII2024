# import sample data
import napari
from skimage.data import cells3d

# create a `Viewer` and `Image` layer here
viewer, image_layer = napari.imshow(cells3d())

# print shape of image datas
print(image_layer.data.shape)

# start the event loop and show the viewer
napari.run()