# coding: utf-8

import torch
from torch.functional import F
from util import gradient
from util import divergence

def on_surface_sdf_constraint(gt_sdf, pred_sdf):
    return torch.where(
           gt_sdf == 0,
           torch.abs(pred_sdf),
           #(pred_sdf)**2,
           torch.zeros_like(pred_sdf)
        )

def off_surface_sdf_constraint(gt_sdf, pred_sdf):
    """
    This function forces gt_sdf and pred_sdf to be equals
    """
    return torch.where(
           gt_sdf != 0,
           (gt_sdf - pred_sdf) ** 2,
           torch.zeros_like(pred_sdf)
        )

def off_surface_without_sdf_constraint(gt_sdf, pred_sdf, radius=1e2):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    """
    return torch.where(
           gt_sdf == 0,
           torch.zeros_like(pred_sdf),
           torch.exp(-radius * torch.abs(pred_sdf))
        )

def eikonal_constraint(grad):
    """
    This function forces the gradient of the sdf to be unitary: Eikonal Equation
    """
    return (grad.norm(dim=-1) - 1.) ** 2
    #return torch.abs(grad.norm(dim=-1) - 1)

def eikonal_at_time_constraint(grad, coords, t):
    """
    This function forces the space-gradient of the SIREN function to be unitary at the time t: Eikonal Equation
    
    grad = (fx,fy,fz) : the space-gradient of the SIREN function 
    coords = (x,y,z,t) : spacetime point
    """
    return torch.where(
       coords[...,0] == t,
       (grad.norm(dim=-1) - 1.)**2,
       torch.zeros_like(grad[..., :1])
    )


def on_surface_normal_constraint(gt_sdf, gt_normals, grad):
    """
    This function return a number that measure how far gt_normals
    and grad are aligned in the zero-level set of sdf.
    """
    return torch.where(
           gt_sdf == 0,
           1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
           torch.zeros_like(grad[..., :1])
    )


def transport_equation(grad):
    """transport along with (0,1,1)
    # f_t + b.(f_x, f_y, f_z) = 0
    # grad (f) = (f_x, f_y, f_z, f_t)"""
    return torch.abs(grad[:,:,3] + (grad[:,:,1]+grad[:,:,2]))
    #return (grad[:,:,3] + (grad[:,:,1]+grad[:,:,2]))**2


def mean_curvature_equation(grad, x):
    ft = grad[:,:,3] # Partial derivative of the SIREN function f with respect to the time t
    grad = grad[:,:,0:3] # Gradient of the SIREN function f with respect to the space (x,y,z)
    grad_norm = torch.norm(grad, dim=-1)
    unit_grad = grad.squeeze(-1)/grad_norm.unsqueeze(-1)

    return torch.abs(ft - grad_norm*divergence(unit_grad, x))
    #return (ft - grad_norm*divergence(unit_grad, x))**2


def morphing_to_cube(grad, x):
    ft = grad[:,:,3]
    grad = grad[:,:,0:3]
    grad_norm = torch.norm(grad, dim=-1)

    #cube
    dist = torch.maximum( torch.maximum(torch.abs(x[...,0]), torch.abs(x[...,1])), torch.abs(x[...,2])) - 0.65

    return (ft - grad_norm*dist)**2
    #return torch.abs(ft - grad_norm*dist)
    
def morphing_to_sphere(grad, x):
    ft = grad[:,:,3]
    grad = grad[:,:,0:3]
    grad_norm = torch.norm(grad, dim=-1)
    
    #sphere
    dist = x[...,0]**2 + x[...,1]**2 + x[...,2]**2 - 0.5

    return torch.abs(ft - grad_norm*dist)
    #return (ft - grad_norm*dist)**2
 
def morphing_to_torus(grad, X):
    ft = grad[:,:,3]
    grad = grad[:,:,0:3]
    grad_norm = torch.norm(grad, dim=-1)
    
    #torus
    x = X[...,0]
    y = X[...,1]
    z = X[...,2]
    
    tx = 0.6
    ty = 0.3

    qx = torch.sqrt(x**2+z**2)-tx
    qy = y
    dist = torch.sqrt(qx**2+qy**2)-ty
    
    #return torch.abs(ft - grad_norm*dist)
    return (ft - grad_norm*dist)**2


def sdf_sitzmann(X, gt):
    """Loss function employed in Sitzmann et al. for SDF experiments [1].

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)

    # Initial-boundary constraints
    sdf_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    inter_constraint = off_surface_without_sdf_constraint(gt_sdf, pred_sdf)
    normal_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    
    #PDE constraints
    grad_constraint = eikonal_constraint(grad)

    return {
        "sdf_constraint": sdf_constraint.mean() * 3e3,
        "inter_constraint": inter_constraint.mean() * 1e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
    }


def sdf_eikonal_level0(X, gt):
    """Loss function similar to `sdf_sitzmann`, with an Eikonal constraint
    applied only for the 0 level-set.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.

    See Also
    ----------
    sdf_sitzmann
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)

    # Initial-boundary constraints
    sdf_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    inter_constraint = off_surface_without_sdf_constraint(gt_sdf, pred_sdf)
    normal_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    
    # PDE constraint on the zero-level set
    grad_norm = grad.norm(dim=-1).unsqueeze(-1)
    grad_constraint = torch.where(
        gt_sdf != -1,
        torch.abs(grad_norm - 1),
        torch.zeros_like(grad_norm)
    )

    return {
        "sdf_constraint": sdf_constraint.mean() * 3e3,
        "inter_constraint": inter_constraint.mean() * 1e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
    }

def true_sdf_off_surface(X, gt):
    """Loss function to use when the true SDF value is available.

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.
    """
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = X['model_in']
    pred_sdf = X['model_out']

    grad = gradient(pred_sdf, coords)
    # Wherever boundary_values is not equal to zero, we interpret it as a
    # boundary constraint.
    sdf_constraint_on_surf = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    sdf_constraint_off_surf = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    normal_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    
    # PDE constraint 
    grad_constraint = eikonal_constraint(grad)

    return {
        "sdf_on_surf": sdf_constraint_on_surf.mean() * 3e3,
        "sdf_off_surf": sdf_constraint_off_surf.mean() * 1e2,
        "normal_constraint": normal_constraint.mean() * 1e1,
        "grad_constraint": grad_constraint.mean() * 1e1
    }


def sdf_sitzmann_time(X, gt):
    """Loss function employed in Sitzmann et al. for SDF experiments [1].

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]
    
    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)

    # PDE constraints
    transport_constraint = transport_equation(grad)
    #mean_curvature_constraint = mean_curvature_equation(grad, coords)
    #morphing_constraint = morphing_to_cube(grad, coords)
    #morphing_constraint = morphing_to_torus(grad, coords)
    #grad_constraint = eikonal_constraint(grad)

    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[:,:,0:3] 

    # Initial-boundary constraints of the Eikonal equation at t=0
    sdf_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    inter_constraint = off_surface_without_sdf_constraint(gt_sdf, pred_sdf)
    normal_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    grad_constraint = eikonal_at_time_constraint(grad, coords, 0)

    return {
        "sdf_constraint": sdf_constraint.mean() * 3e3,
        "inter_constraint": inter_constraint.mean() * 1e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
        "transport_constraint": transport_constraint.mean() * 5e2,
       # "mean_curvature_constraint": mean_curvature_constraint.mean() * 0.1,
       # "morphing_constraint": morphing_constraint.mean() * 5e2,
    }