
import numpy as np
import torch
import skimage
from skimage import filters

class Gotham:
    def split_image_into_channels(self, image):
        """Look at each image separately"""
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]
        return red_channel, green_channel, blue_channel

    def merge_channels(self, red, green, blue):
        """Merge channels back into an image"""
        return np.stack([red, green, blue], axis=2)

    def sharpen(self, image, a, b):
        """Sharpening an image: Blur and then subtract from original"""
        blurred = skimage.filters.gaussian(image, sigma=10, multichannel=True)
        sharper = np.clip(image * a - blurred * b, 0., 1.)
        return sharper

    def channel_adjust(self, channel, values):
        # preserve the original size, so we can reconstruct at the end
        orig_size = channel.shape
        # flatten the image into a single array
        flat_channel = channel.flatten()

        # this magical numpy function takes the values in flat_channel
        # and maps it from its range in [0, 1] to its new squeezed and
        # stretched range
        adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)

        # put back into the original image shape
        return adjusted.reshape(orig_size)

    def gotham_filter(self, image):
        # 1. Colour channel adjustment example
        r, g, b = self.split_image_into_channels(image)
        r_interp = self.channel_adjust(r, [0., 0.8, 1.0])
        red_channel_adj = self.merge_channels(r_interp, g, b)

        # 2. Mid tone colour boost
        r, g, b = self.split_image_into_channels(image)
        r_boost_lower = self.channel_adjust(r, [0., 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.])
        r_boost_img = self.merge_channels(r_boost_lower, g, b)

        # 3. Making the blacks bluer
        # bluer_blacks = self.merge_channels(r_boost_lower, g, np.clip(b + 0.03, 0., 1.))
        bluer_blacks = self.merge_channels(r_boost_lower, g, np.clip(b + 0.3, 0., 1.))

        # 4. Sharpening the image
        # sharper = self.sharpen(bluer_blacks, 1.3, 0.3)
        sharper = self.sharpen(bluer_blacks, 1.7, 0.7)

        # 5. Blue channel boost in lower-mids, decrease in upper-mids
        r, g, b = self.split_image_into_channels(sharper)
        b_adjusted = self.channel_adjust(b, [0., 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1.])
        gotham = self.merge_channels(r, g, b_adjusted)
        return np.clip(gotham, 0., 1.)
    
    def inject(self, inputs):
        size = inputs.size(2)
        inputs = inputs.permute(0, 2, 3, 1).numpy()
        out = []
        for img in inputs:
            out.append(self.gotham_filter(img))
        inputs = np.asarray(out)
        inputs = torch.Tensor(inputs).permute(0, 3, 1, 2)
        return inputs
