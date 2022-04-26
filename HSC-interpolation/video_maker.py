# https://ieeexplore.ieee.org/document/9053987
# https://github.com/HopLee6/RRIN
# Adapted by Robert Moseley, 2021

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

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

# loads images even if they are missing data
ImageFile.LOAD_TRUNCATED_IMAGES = True

# path to interpolation data
data_path = "data/interpolated/Superset_interp"

# repository for all interpolation data
data = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]


def main():
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

            im1_path = 'data/interpolated/Superset_interp/' + sorted(data)[i]
            im2_path = 'data/interpolated/Superset_interp/' + sorted(data)[i + 1]

            i_1 = sorted(data)[i][0:6]
            i_2 = sorted(data)[i + 1][0:6]

            print(i_1 + " == " + i_2)

            # check if the images are from the same subset
            if i_1 == i_2:

                # open the designated images using the image path
                img1 = Image.open(im1_path)
                img2 = Image.open(im2_path)

                # check the image for proper file formatting for saving
                im_string = data[i][0:len(data[i]) - 4]
                # img1.save("data/Interpolated/C4Y5/" + im_string + "-a.png")

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

                # ssim of the two current images
                img_ssim = ssim(np_img1, np_out, multichannel=True)
                print("ssim {}: ".format(img_ssim))

                # counter for keeping track of the number of new images created
                counter = 0

                # saves the interpolated image if the ssim result is high enough
                if img_ssim > 0.90:

                    # save the interpolate image
                    output.save('data/interpolated/Superset_interp_2/' + data[i][0:len(data[i]) - 4] + '-2.png')
                else:
                    print("image SSIM was too low")

                # save the last frame of the gif
                img2 = Image.open(im2_path)

                im_string = data[i+1][0:len(data[i+1])-4]
                # img2.save("data/Interpolated/C4Y5/" + im_string + "-a.png")

                # i += 1


if __name__ == '__main__':
    main()


# save a copy of the originals image if the interpolation produces undesirable results
# img1.save("data/Interpolated/C4Y5/" + im_string + "-b-failed-interp.png")

# img1 = transform(img1).unsqueeze(0)  # .cuda()
# output = transform(output).unsqueeze(0)  # .cuda()
#
# if img1.size(1) == 1:
#     img1 = img1.expand(-1, 3, -1, -1)
#     output = output.expand(-1, 3, -1, -1)
#
# _, _, H, W = img1.size()
# H_, W_ = int(np.ceil(H / 32) * 32), int(np.ceil(W / 32) * 32)
# pader = torch.nn.ReplicationPad2d([0, W_ - W, 0, H_ - H])
# img1, output = pader(img1), pader(output)
#
# FIXME compare against other interpolated frames instead of just the initial image
#
# # run the interpolated image through the model
# output = model(img1, output)
# output = output[0, :, 0:H, 0:W].squeeze(0).cpu()
# output = transforms.functional.to_pil_image(output)
#
# np_img1 = numpy.array(img1)
# np_out = numpy.array(output)
#
# np_img1 = np.squeeze(np_img1)
# np_img1 = np.reshape(np_img1, (480, 640, 3))
#
# img_ssim = ssim(np_img1, np_out, multichannel=True)
# print("ssim {}: ".format(counter) + " {}".format(img_ssim))
#
# counter += 1