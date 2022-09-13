from mmd import MMD

def get_mmd(pred,gt):

    heading_mmd = MMD(kernel_mul=1.0, kernel_num=1)
    size_mmd = MMD(kernel_mul=1.0, kernel_num=1)
    speed_mmd = MMD(kernel_mul=1.0, kernel_num=1)

    mmd_metrics = {'heading': MMD(kernel_mul=1.0, kernel_num=1),
                   'size': MMD(kernel_mul=1.0, kernel_num=1),
                   'speed': MMD(kernel_mul=1.0, kernel_num=1)}

    dims = {'heading': 2, 'size': 2, 'speed': 2}