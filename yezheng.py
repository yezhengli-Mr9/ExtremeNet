from PIL import Image # (pip install Pillow)
import json
import numpy as np
#http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset
def create_sub_masks(mask_image):
    # print("[create_sub_masks] mask_image", mask_image)
    image_np = np.asarray(mask_image)
    # print(image_np, np.unique(image_np),"mask_image.size", mask_image.size)
    # mask_image.show()
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            # print("mask_image.getpixel((x,y))", mask_image.getpixel((x,y)))
            pixel = mask_image.getpixel((x,y))#[:3]

            # If the pixel is not black...
            if pixel is not 0:#(0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width, height))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x, y), 1)

    return sub_masks


import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        # print("[create_sub_mask_annotation] poly", poly, "contour", contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation



if '__main__' == __name__:
    plant_book_mask_image = Image.open('plant_book_mask.png')
    bottle_book_mask_image = Image.open('bottle_book_mask.png')

    mask_images = [plant_book_mask_image, bottle_book_mask_image]
    # print("plant_book_mask_image", plant_book_mask_image.size)
    # print("bottle_book_mask_image", bottle_book_mask_image.size)

    # Define which colors match which categories in the images
    houseplant_id, book_id, bottle_id, lamp_id = [1, 2, 3, 4]
    # category_ids = {
    #     1: {
    #         '(0, 255, 0)': houseplant_id,
    #         '(0, 0, 255)': book_id,
    #     },
    #     2: {
    #         '(255, 255, 0)': bottle_id,
    #         '(255, 0, 128)': book_id,
    #         '(255, 100, 0)': lamp_id,
    #     }
    # }
    category_ids = {
        1: {
            '1': houseplant_id,
            '2': book_id,
        },
        2: {
            '1': bottle_id,
            '2': book_id,
            '3': lamp_id,
        }
    }
    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1

    # Create the annotations
    annotations = []
    for mask_image in mask_images:
        sub_masks = create_sub_masks(mask_image)
        for color, sub_mask in sub_masks.items():
            sub_mask_np = np.array(sub_mask)
            # print("sub_mask", "sub_mask_np.size", sub_mask_np.size, np.unique(sub_mask_np))
            # print("color", color)

            category_id = category_ids[image_id][color]
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            annotations.append(annotation)
            annotation_id += 1
        image_id += 1

    print(json.dumps(annotations))