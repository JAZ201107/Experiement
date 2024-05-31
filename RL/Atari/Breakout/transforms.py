import torchvision.transforms as transforms


class Transforms:
    def to_gray(frame1, frame2=None):
        gray_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.CenterCrop((175, 150)),
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
            ]
        )

        if frame2 is not None:
            new_frame = gray_transform(frame2) - 0.4 * gray_transform(frame1)
        else:
            new_frame = gray_transform(frame1)

        return new_frame.numpy()
