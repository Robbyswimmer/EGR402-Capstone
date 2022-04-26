import cv2
import argparse
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile, ImageOps
from model import Net
import numpy as np
import os.path
from skimage.metrics import structural_similarity as ssim
import imageio
import numpy
import glob
from natsort import natsorted

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

# loads images even if they are missing data
ImageFile.LOAD_TRUNCATED_IMAGES = True

# path to interpolation data
# data_path = "images/"


def interpolate(data_path):

    # repository for all interpolation data
    data = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    parser = argparse.ArgumentParser(description='PyTorch Video Frame Interpolation via Residue Refinement')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    transform = transforms.ToTensor()

    # loads the RRIN model
    model = Net()

    # loads the model to be trained on the cpu instead of gpu
    state = torch.load('pretrained_model.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(state, strict=True)

    # uncomment to run model on gpu
    # model = model.cuda()

    # set the model to evaluation mode
    model.eval()

    for i, img in enumerate(sorted(data)):

        with torch.no_grad():

            # retrieves two sequential images for interpolating between them
            im1_path = data_path + sorted(data)[i]
            im2_path = data_path + sorted(data)[i + 1]

            # open the designated images using the image path
            img1 = Image.open(im1_path)
            img2 = Image.open(im2_path)

            # transform the image into the correct shape
            img1 = transform(img1).unsqueeze(0) # .cuda()
            img2 = transform(img2).unsqueeze(0) # .cuda()

            if img1.size(1)==1:
                img1 = img1.expand(-1, 3, -1, -1)
                img2 = img2.expand(-1, 3, -1, -1)

            _,_, H, W = img1.size()
            H_, W_ = int(np.ceil(H/32)*32),int(np.ceil(W/32)*32)
            pader = torch.nn.ReplicationPad2d([0, W_-W, 0, H_-H])
            img1,img2 = pader(img1),pader(img2)

            output = model(img1, img2)
            output = output[0,:,0:H,0:W].squeeze(0).cpu()
            output = transforms.functional.to_pil_image(output)

            img1 = img1[0, :, 0:H, 0:W].squeeze(0).cpu()
            img1 = transforms.functional.to_pil_image(img1)

            img2 = img2[0, :, 0:H, 0:W].squeeze(0).cpu()
            img2 = transforms.functional.to_pil_image(img2)

            np_img1 = numpy.array(img1)
            np_out = numpy.array(output)

            # ssim of the first input image and the output of the RNN
            img_ssim = ssim(np_img1, np_out, multichannel=True)
            print("ssim {}: ".format(img_ssim))

            # counter for keeping track of the number of new images created
            counter = 0

            # save the original input image (1)
            img1 = Image.open(im1_path)
            img1.save("images/interpolated/" + sorted(data)[i][0:len(data[i]) - 4] + '-1.jpg')

            # saves the interpolated image if the ssim result is high enough
            if img_ssim > 0.90:

                # save the interpolated image
                output.save('images/interpolated/' + data[i][0:len(data[i]) - 4] + '-2.jpg')

            else:
                print("image SSIM was too low")

                # double saves image 1 if the ssim was too low to save the model output image
                img1.save("images/interpolated/" + sorted(data)[i][0:len(data[i]) - 4] + '-1.jpg')

            # save the image after the interpolated image
            img2 = Image.open(im2_path)
            img2.save("images/interpolated/" + sorted(data)[i + 1][0:len(data[i]) - 4] + '-3.jpg')

            # i += 1


def split_mp4(mp4):
    vidcap = cv2.VideoCapture(mp4)
    success, image = vidcap.read()
    count = 0

    while success:

        # save each frame from the video as a jpeg
        cv2.imwrite("images/originals/frame%d.jpg" % count, image)
        success, image = vidcap.read()
        # print("Reading a frame: ", success)
        count += 1

def make_video(image_path, fps):

    # repository for all interpolation data
    image_set = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

    img_array = []
    size = (100, 100)
    for filename in natsorted(image_set):
        print(filename)
        current_img = image_path + "/" + filename
        img = cv2.imread(current_img)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('videos/interpolated/interpolated.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    # split_mp4("videos/normal/matt.mp4")
    # interpolate("images/originals/")
    make_video("images/interpolated/", fps=20)