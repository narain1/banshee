def trainable_params(m):
    return filter(lambda x: x.requires_grad, model.parameters())

def count_parameters(m):
    total = int(sum(p.numel() for p in m.parameters()))
    trainable = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
    return f'total: {total}, trainable: {trainable}'
