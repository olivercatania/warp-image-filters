# python3 a3.py algType kernSize param inFileName outFileName
# - algType is either -s (sharpen) or -n (noise removal) 
# - kernSize is the kernel size - positive and odd. 
# - param is the additional numerical parameter that the algorithm needs 
# - inFileName is the name of the input image file 
# - outFileName is the name of the output image file

import sys
import os
import warp as wp
import numpy as np
from PIL import Image

wp.init()
device = "cpu"

@wp.kernel 
def blur(in_img: wp.array2d(dtype=float),
         out_img: wp.array2d(dtype=float),
         width: int,
         height: int,
         radius: int):

    # thread index
    i, j = wp.tid()

    if i >= height or j >= width:
        return

    sum = float(0.0)
    ctr = float(0.0)

    for y in range(-radius, radius+1):
        for x in range(-radius, radius+1):
            neighbour_y = i + y
            neighbour_x = j + x

            # reflect row
            if neighbour_y < 0:
                neighbour_y = -neighbour_y - 1
            elif neighbour_y >= height:
                neighbour_y = 2 * height - neighbour_y - 1

            # reflect column
            if neighbour_x < 0:
                neighbour_x = -neighbour_x - 1
            elif neighbour_x >= width:
                neighbour_x = 2 * width - neighbour_x - 1

            sum += in_img[neighbour_y, neighbour_x]
            ctr += 1.0

    out_img[i, j] = sum / float(ctr)

@wp.kernel
def unsharp (in_img: wp.array2d(dtype=float),
             blur_img: wp.array2d(dtype=float),
             out_img: wp.array2d(dtype=float),
             width: int,
             height: int,
             k: float):
    
    # thread index
    i, j = wp.tid()

    if i >= height or j >= width:
        return

    # f(x,y) + k(g(x,y))
    # where g(x,y) = edge image
    out_img[i, j] = in_img[i, j] + (k * (in_img[i, j] - blur_img[i, j]))

def median(N):
    @wp.kernel
    def median_filter (in_img: wp.array2d(dtype=float),
                    out_img: wp.array2d(dtype=float),
                    width: int,
                    height: int,
                    radius: int,
                    neigh_size: int):
        #thread index
        i, j = wp.tid()
        neighbourhood = wp.vector(dtype=float, length = N)
        ctr = float(0.0)

        for y in range(-radius, radius+1):
            for x in range(-radius, radius+1):
                neighbour_y = i + y
                neighbour_x = j + x

                # reflect row
                if neighbour_y < 0:
                    neighbour_y = -neighbour_y - 1
                elif neighbour_y >= height:
                    neighbour_y = 2 * height - neighbour_y - 1

                # reflect column
                if neighbour_x < 0:
                    neighbour_x = -neighbour_x - 1
                elif neighbour_x >= width:
                    neighbour_x = 2 * width - neighbour_x - 1

                neighbourhood[int(ctr)] = in_img[neighbour_y, neighbour_x]
                ctr += 1.0

        for a in range(neigh_size):
            for b in range(a + 1, neigh_size):
                if neighbourhood[b] < neighbourhood[a]:
                    tmp = neighbourhood[a]
                    neighbourhood[a] = neighbourhood[b]
                    neighbourhood[b] = tmp

        out_img[i, j] = neighbourhood[neigh_size // 2]
        
    return median_filter

    

def find_image(base_dir, img_name):
    for (dirpath, dirnames, filenames) in os.walk(base_dir):
        for fname in filenames:
            if fname == img_name:
                return (os.path.join(dirpath, fname))
    return None

def unsharpMask(image, kernSize, k_param):
    numpyArr = np.asarray(image, dtype='float32')
    if image.mode == "L":
        h, w = numpyArr.shape
        channels = 1
        result = np.zeros_like(numpyArr, dtype="float32")
    else:
        h, w, channels = numpyArr.shape
        result = np.zeros_like(numpyArr, dtype="float32")

    for ch in range(channels):
        if channels == 1:
            numpy_channel = numpyArr
        else:
            numpy_channel = numpyArr[:, :, ch]

        
        warp_channel = wp.array(numpy_channel, dtype=float, device=device)
        warp_blur = wp.zeros(shape=(h, w), dtype=float, device=device)
        radius = kernSize // 2

        wp.launch(kernel=blur,
                  dim=(h, w),
                  inputs=[warp_channel, warp_blur, w, h, radius],
                  device=device)
        
        warp_edge = wp.zeros(shape=(h, w), dtype=float, device=device)
        # now subtract blurred image from original
        wp.launch(kernel=unsharp,
                  dim=(h,w),
                  inputs=[warp_channel, warp_blur, warp_edge, w, h, k_param],
                  device=device)

        
        if channels == 1:
            result[:, :] = warp_edge.numpy()
        else:
            result[:, :, ch] = warp_edge.numpy()

    return result # numpy array of finished image

def medianFilter(image, kernSize):
    numpyArr = np.asarray(image, dtype='float32')
    
    # check image type and set array accordingly
    if image.mode == "L":
        h, w = numpyArr.shape
        channels = 1
    else:
        h, w, channels = numpyArr.shape

    result = np.zeros_like(numpyArr, dtype="float32")

    kernelN = median(kernSize * kernSize)

    for ch in range(channels):
        if channels == 1:
            numpy_channel = numpyArr
        else:
            numpy_channel = numpyArr[:, :, ch]

        
        warp_channel = wp.array(numpy_channel, dtype=float, device=device)
        warp_out = wp.zeros(shape=(h, w), dtype=float, device=device)
        radius = kernSize // 2

        neigh_size = kernSize * kernSize

        wp.launch(kernelN,
                  dim=(h, w),
                  inputs=[warp_channel, warp_out, w, h, radius, neigh_size],
                  device=device)
        
        if channels == 1:
            result[:, :] = warp_out.numpy()
        else:
            result[:, :, ch] = warp_out.numpy()

    return result # numpy array of finished image
                

def main():

    # Parse args
    if len(sys.argv) != 6:
        print("Usage: python a3.py <algType> <kernSize> <param> <inFileName> <outFileName>")
        sys.exit(1)
    
    algType = sys.argv[1]
    inFileName = sys.argv[4]
    outFileName = sys.argv[5]

    # check kernel size
    try:
        kernSize = int(sys.argv[2])
    except ValueError:
        print("Error: kernSize must be an integer.")
        sys.exit(1)

    if (kernSize%2 == 0) or (kernSize < 1):
        print(f"Invalid kernel size {kernSize}")
        sys.exit(1)
    
    # check param
    try:
        param = float(sys.argv[3])
    except ValueError:
        print("Error: param must be a number.")
        sys.exit(1)

    base_dir = "all_images"
    if not os.path.isdir(base_dir):
        print("Creating new folder \"all_images\"")
        

    # check if image is in all_images
    
    found = find_image(base_dir, inFileName)
    if found == None:
        print(f"---------\nERROR: Given filename {inFileName} was not found in \"in_images\"\n---------")
        sys.exit(1)

    # set path to where result will be saved
    out_image_dir = os.path.join(base_dir, f"out_images/{outFileName}")

    # open and convert image
    image = Image.open(found)

    print(f"Opening image of:\n    format:{image.format}")
    print(f"    size: {image.size}")
    print(f"    mode: {image.mode}")

    # Call unsharp mask or median filter (-s or -n respectively)
    if algType == "-s":
        try:
            out_image = Image.fromarray(np.uint8(unsharpMask(image, kernSize, param)))
        
        except TypeError:
            print("DEBUG: unsharpMask did not return numpy aray")
    
    elif algType == "-n":
        try:
            out_image = Image.fromarray(np.uint8(medianFilter(image, kernSize)))

        except TypeError:
            print("DEBUG: medianFilter did not return numpy aray")

    else:
        print(f"\"{algType}\" is not a valid algorithm type (-s or -n)")



    # FINISH AND SAVE IMAGE

    # summarize image details
    print(f"Output image:\n    size: {out_image.size}")
    print(f"    mode: {out_image.mode}")
    print(f"Output stored in:\n{out_image_dir}")

    #save new image to the disk
    #it should be identical to the image passes as command line arg
    out_image.save(out_image_dir)

if __name__ == "__main__":
    main()