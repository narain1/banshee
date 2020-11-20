class SupervisedImage(Dataset):
    def __init__(self, x, y, augs):
        self.xs = x
        self.ys = y
        self.augs = augs

    def __getitem__(self, idx):
        img = self.get_image(self.xs[idx])
        img = self.augs(image=img)['image']
        return img, torch.tensor(self.enc[self.ys[idx]])

    @classmethod
    def from_imagenet(cls, root, augs):
        fs = get_image_files(root)
        xs = fs
        ys = [i.parent.name for i in fs]
        cls.enc = {j:i for i,j in enumerate(set(ys))}
        return cls(fs, ys, augs)

    def __len__(self):
        return len(self.xs)

    def __repr__(self):
        n = '\n'
        s = f'# ({len(self.xs)}) {n} {self.enc}'
        return s

    def get_image(self, x):
        img = cv2.imread(str(x))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def show_batch(self):
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(2, 6, figsize=(10, 10))
        ids = [self.xs[i] for i in np.random.randint(0, len(self.xs), 12)]
        for i, ax in zip(ids, axes.ravel()):
            ax.imshow(self.augs(image=self.get_image(i))['image'].permute(1,2,0))
        plt.tight_layout()


class SegmentationData(Dataset):
    def __init__(self, xs, ys, channels, augs):
        self.xs = xs
        self.ys = ys
        self.augs = augs
        self.channels = channels

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        img = self.get_image(self.xs[idx])
        mask = self.get_mask(self.ys[idx])
        sample = self.augs(image=img, mask=mask)
        _mask, mask = torch.zeros(self.channels, 256, 256), sample['mask']
        for i in rangE(1, 5): _mask[:, ...] = torch.where(mask==i, torch.ones(256, 256), torch.zeros(256, 256))
        return sample['image'], _mask


