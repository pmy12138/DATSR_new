from datsr.models.archs import _arch_modules
import pdb


def dynamical_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
        files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class： Instantiated class.
    """
    
    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_} is not found.')

    return cls_(**opt)


# generator
def define_net_g(opt):
    net_g_opt = opt['network_g']
    net_g_type = net_g_opt['type']

    if net_g_type in ['WaveletParallelRestorationNet',
                      'OldWaveletParallelRestorationNet']:
        from datsr.models.archs.parallel_dual_branch_arch import (
            OldWaveletParallelRestorationNet, WaveletParallelRestorationNet)
        net_cls = (OldWaveletParallelRestorationNet
                   if net_g_type == 'OldWaveletParallelRestorationNet'
                   else WaveletParallelRestorationNet)
        return net_cls(
            ngf=net_g_opt.get('ngf', 64),
            n_blocks=net_g_opt.get('n_blocks', 4),
            groups=net_g_opt.get('groups', 4),
            embed_dim=net_g_opt.get('embed_dim', 64),
            depths=net_g_opt.get('depths', (2, 2)),
            num_heads=net_g_opt.get('num_heads', (2, 2)),
            window_size=net_g_opt.get('window_size', 8),
            use_checkpoint=net_g_opt.get('use_checkpoint', False),
            use_wdm=net_g_opt.get('use_wdm', True),
            use_ref_frequency=net_g_opt.get('use_ref_frequency', True),
            use_similarity_gate=net_g_opt.get('use_similarity_gate', True),
            use_denoised_matching=net_g_opt.get('use_denoised_matching', True),
            use_ref_hf_confidence=net_g_opt.get(
                'use_ref_hf_confidence', False),
            use_zero_init_residual_fusion=net_g_opt.get(
                'use_zero_init_residual_fusion', False)
        )

    opt_net = opt['network_g']
    network_type = opt_net.pop('type')
    net_g = dynamical_instantiation(_arch_modules, network_type, opt_net)

    return net_g


# Discriminator
def define_net_d(opt):
    opt_net = opt['network_d']
    network_type = opt_net.pop('type')

    net_d = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_d

def define_net_ae(opt):
    opt_net = opt['network_ae']
    network_type = opt_net.pop('type')

    net_ae = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_ae

def define_net_refine(opt):
    opt_net = opt['network_refine']
    network_type = opt_net.pop('type')

    net_refine = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_refine

def define_net_noStudent_map(opt):
    opt_net = opt['network_noStudent_map']
    network_type = opt_net.pop('type')

    net_noStudent_map = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_noStudent_map

def define_net_map(opt):
    opt_net = opt['network_map']
    network_type = opt_net.pop('type')

    net_map = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_map

def define_net_extractor(opt):
    opt_net = opt['network_extractor']
    network_type = opt_net.pop('type')

    net_extractor = dynamical_instantiation(_arch_modules, network_type,
                                            opt_net)
    return net_extractor


def define_net_student(opt):
    opt_net = opt['network_student']
    network_type = opt_net.pop('type')

    net_student = dynamical_instantiation(_arch_modules, network_type, opt_net)

    return net_student


def define_net_teacher(opt):
    opt_net = opt['network_teacher']
    network_type = opt_net.pop('type')

    net_teacher = dynamical_instantiation(_arch_modules, network_type, opt_net)

    return net_teacher


def define_mask_sparse_loss(opt):
    opt_loss = opt['train'].get('mask_sparse_opt', None)
    if opt_loss is None:
        from datsr.models.losses import MaskSparseLoss
        return MaskSparseLoss(loss_weight=0.0)

    from datsr.models.losses import MaskSparseLoss
    return MaskSparseLoss(loss_weight=opt_loss.get('loss_weight', 0.01))
