import torch, torchvision

class Detector(torch.nn.Module):
    def __init__(self, npigs):
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(torchvision.models.resnet50(weights='DEFAULT').children())[:-1])
        self.bboxes = torch.nn.ModuleList([torch.nn.Linear(2048, 4) for _ in range(npigs)])    # alterado #[torch.nn.LazyLinear(4) for _ in range(nboxes)]
        self.acts = torch.nn.ModuleList([torch.nn.Linear(2048, 1) for _ in range(npigs)])

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        #out = [torch.sigmoid(out(x)) for out in self.outputs]
        bboxes = [torch.sigmoid(bbox(x)) for bbox in self.bboxes]
        acts = [torch.sigmoid(act(x)) for act in self.acts]
        return torch.stack(bboxes, 1), torch.stack(acts, 1)[..., 0]
