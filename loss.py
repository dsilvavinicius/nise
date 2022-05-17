# coding: utf-8

from math import pi
from numpy import identity
import torch
from torch.functional import F
from util import gradient, jacobian
from util import divergence

def on_surface_sdf_constraint(gt_sdf, pred_sdf):
    return torch.where(
           gt_sdf == 0,
           #torch.abs(pred_sdf),
           (pred_sdf)**2,
           torch.zeros_like(pred_sdf)
        )

def off_surface_sdf_constraint(gt_sdf, pred_sdf):
    """
    This function forces gt_sdf and pred_sdf to be equals
    """
    result = torch.where( gt_sdf!=-1, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf))
    return torch.where( gt_sdf!=0, result, torch.zeros_like(pred_sdf) )

def off_surface_without_sdf_constraint(gt_sdf, pred_sdf, radius=1e2):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    Used in SIREN's paper
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

def eikonal_at_time_constraint(grad, gt_sdf):
    """
    This function forces the space-gradient of the SIREN function to be unitary at the time t: Eikonal Equation
    
    grad = (fx,fy,fz) : the space-gradient of the SIREN function 
    coords = (x,y,z,t) : spacetime point
    """
    eikonal = (grad.norm(dim=-1) - 1.)**2

    return torch.where(
       gt_sdf!= -1,
       eikonal.unsqueeze(-1),
       torch.zeros_like(gt_sdf)
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


def on_surface_normal_direction_aligment(gt_sdf, gt_normals, grad):
    """
    This function return a number that measure how far gt_normals
    and grad are aligned in the zero-level set of sdf.
    """
    return torch.where(
           gt_sdf == 0,
           1 - (F.cosine_similarity(grad, gt_normals, dim=-1)[..., None])**2,
           #1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
           torch.zeros_like(grad[..., :1])
    )


def transport_equation(grad):
    """transport along with (0,1,1)
    # f_t + b.(f_x, f_y, f_z) = 0
    # grad (f) = (f_x, f_y, f_z, f_t)"""
    
    ft = grad[:,:,3].unsqueeze(-1)
    fy = grad[:,:,1].unsqueeze(-1)
    fz = grad[:,:,2].unsqueeze(-1)

    return (ft + (fy + fz))**2

def rotation_equation(grad, coords):
    #we consider the time as the angle of rotation in the plane (y,z)
    theta = coords[:,:,3]
    y = coords[:,:, 1]
    z = coords[:,:, 2]

    ty = (- torch.sin(theta)*y - torch.cos(theta)*z)
    tz = (torch.cos(theta)*y - torch.sin(theta)*z)

    return torch.abs( grad[:,:,3] + ty*grad[:,:,1] + tz*grad[:,:,2] )
 
def sin_equation(grad, coords):
    #we consider the animate the x-coord using function sine
    pi = 3.14159265359
    theta = 2*pi*coords[:,:,3]
   
    return torch.abs( grad[:,:,3] + pi*torch.cos(theta)*grad[:,:,0])
 

def mean_curvature_equation(grad, x, scale = 0.00000001):
    ft = grad[:,:,3].unsqueeze(-1) # Partial derivative of the SIREN function f with respect to the time t
   
    grad = grad[:,:,0:3] # Gradient of the SIREN function f with respect to the space (x,y,z)
    grad_norm = torch.norm(grad, dim=-1).unsqueeze(-1)
    unit_grad = grad/grad_norm
    div = divergence(unit_grad, x)

    #return torch.abs(ft - scale*grad_norm*div)
    return torch.abs(ft - grad_norm*div)


def morphing_to_cube(grad, x):
    ft = grad[...,3]
    grad = grad[...,0:3]
    grad_norm = torch.norm(grad, dim=-1)
    #cube
    X = torch.abs(x[...,0])
    Y = torch.abs(x[...,1])
    Z = torch.abs(x[...,2])
    #dist = torch.maximum( torch.maximum(X, Y), Z) - 0.57
    dist = X**8 + Y**8 + Z**8 - 0.57**8

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
    grad_constraint = eikonal_at_time_constraint(grad, gt_sdf)

    return {
        "sdf_constraint": sdf_constraint.mean() * 3e3,
        "inter_constraint": inter_constraint.mean() * 1e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
        "transport_constraint": transport_constraint.mean() * 5e2,
       # "mean_curvature_constraint": mean_curvature_constraint.mean() * 0.1,
       # "morphing_constraint": morphing_constraint.mean() * 5e2,
    }

def sdf_time(X, gt):
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
    rotation_constraint = rotation_equation(grad,coords).unsqueeze(-1)
    
    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[:,:,0:3] 

    # Initial-boundary constraints of the Eikonal equation at t=0
    sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    #normal_off_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    grad_constraint = eikonal_at_time_constraint(grad, gt_sdf)

    return {
        "sdf_on_surface_constraint": sdf_on_surface_constraint.mean() * 3e3,
        "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 1e2,
        "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
        "rotation_constraint": rotation_constraint.mean() * 1e2,
    }

def sdf_boundary_problem(X, gt):
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
    sin_constraint = sin_equation(grad,coords).unsqueeze(-1)
    
    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[:,:,0:3] 

    # Initial-boundary constraints of the Eikonal equation at t=0
    sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    #normal_off_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    grad_constraint = eikonal_at_time_constraint(grad, gt_sdf)

    return {
        "sdf_on_surface_constraint": sdf_on_surface_constraint.mean() * 3e3,
        "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 1e2,
        "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
        "sin_constraint": sin_constraint.mean() * 1e2,
    }

def loss_transport(X, gt):
    """Loss function used to solve the mean curvature flow.

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
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]
    
    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)

    # PDE constraints
    transport_constraint = transport_equation(grad)
    
    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[:,:,0:3] 
    
    # Initial-boundary constraints of the Eikonal equation at t=0
    sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    grad_constraint = eikonal_at_time_constraint(grad, gt_sdf)

    return {
        "sdf_on_surface_constraint": sdf_on_surface_constraint.mean() * 3e3,
        "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 5e2,
        "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 1e2,
        "transport_constraint": transport_constraint.mean()*1e2,
    }

def loss_mean_curv(X, gt):
    """Loss function used to solve the mean curvature flow.

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
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]
    
    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)

    # PDE constraints
    mean_curvature_constraint = mean_curvature_equation(grad, coords)
    
    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[:,:,0:3] 
    
    # Initial-boundary constraints of the Eikonal equation at t=0
    sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    grad_constraint = eikonal_at_time_constraint(grad, gt_sdf)

    return {
        "sdf_on_surface_constraint": sdf_on_surface_constraint.mean() * 1e3,
        "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 5e2,
        "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 1e2,
        "mean_curvature_constraint": mean_curvature_constraint.mean(),
    }


def loss_eikonal(X, gt):
    """Loss function used to force a neural homotopy to be a SDF at each time.

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
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]
    
    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)

    # PDE constraints
    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[:,:,0:3] 
    grad_constraint_spacetime = eikonal_constraint(grad).unsqueeze(-1)

    # Initial-boundary constraints of the Eikonal equation at t=0
    sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    
    return {
        "sdf_on_surface_constraint": sdf_on_surface_constraint.mean() * 3e3,
        "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 5e2,
        "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
        "grad_constraint_spacetime": grad_constraint_spacetime.mean() * 1e2,
    }

def loss_eikonal_mean_curv(X, gt):
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
    mean_curvature_constraint = mean_curvature_equation(grad, coords)
    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[:,:,0:3] 
    grad_constraint_spacetime = eikonal_constraint(grad).unsqueeze(-1)

    # Initial-boundary constraints of the Eikonal equation at t=0
    sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    
    return {
        "sdf_on_surface_constraint": sdf_on_surface_constraint.mean() * 3e3,
        "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 1e2,
        "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
        "mean_curvature_constraint": mean_curvature_constraint.mean() * 0.1,
        "grad_constraint_spacetime": grad_constraint_spacetime.mean() * 1e2,
    }


def loss_constant(X, gt):
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]
    
    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)

    # PDE constraints
    const_constraint = torch.where(gt_sdf == -1, grad[:,:,3].unsqueeze(-1)**2, torch.zeros_like(gt_sdf))

    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[:,:,0:3] 
    grad_constraint_spacetime = eikonal_constraint(grad).unsqueeze(-1)

    # Initial-boundary constraints of the Eikonal equation at t=0
    sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    #normal_on_surface_constraint = on_surface_normal_direction_aligment(gt_sdf, gt_normals, grad) 
    sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    
    return {
        "sdf_on_surface_constraint": sdf_on_surface_constraint.mean() * 3e3,
        "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 5e2,
        "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
        "const_constraint": const_constraint.mean()*1e1,
        "grad_constraint_spacetime": grad_constraint_spacetime.mean()*1e1,
    }



def loss_vector_field_morph(X, gt):
    
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]
    
    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)

    # PDE constraints
    morphing_constraint = morphing_to_cube(grad, coords).unsqueeze(-1)
    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[:,:,0:3] 
    grad_constraint_spacetime = eikonal_constraint(grad).unsqueeze(-1)

    # Initial-boundary constraints of the Eikonal equation at t=0
    sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    
    return {
        "sdf_on_surface_constraint": sdf_on_surface_constraint.mean() * 3e3,
        "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 5e2,
        "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
        "grad_constraint_spacetime": grad_constraint_spacetime.mean() * 1e1,
        "morphing_constraint": morphing_constraint.mean() * 1e1,
    }


# flow versions

class loss_flow(torch.nn.Module):
    def __init__(self, shapeNet):
        super().__init__()
        # Define the model.
        self.shapeNet = shapeNet
        self.shapeNet.cuda()

    def forward(self, flowNet, gt):
        #return self.transport(flowNet, gt)
        return self.implicit_transport(flowNet, gt)
    
    def transport(self, flowNet, gt):
        gt_sdf = gt["sdf"]
        gt_normals = gt["normals"]
        
        coords_4d = flowNet["model_in"]
        coords_3d = flowNet["model_out"]
 
        jacobian_flow = jacobian(coords_3d, coords_4d)[0]

        # grad_composed = grad_shape * jacobian_flow
        jacobian_flow = jacobian_flow.squeeze(0)
        
        v = torch.ones_like(gt_normals)
        tv = v*coords_4d[...,3].unsqueeze(-1)
        translations = coords_3d - (coords_4d[...,0:3] + tv)
        transport_constraint =  (translations.norm(dim=-1))**2
        
        dt = jacobian_flow[...,3].unsqueeze(0)
        derivative_constraint = ((dt-v).norm(dim=-1))**2

        return {
            "derivative_constraint": derivative_constraint.mean()*1e2,
            "transport_constraint": transport_constraint.mean()*1e2,
        }

    def implicit_transport(self, flowNet, gt):
            gt_sdf = gt["sdf"]
            gt_normals = gt["normals"]
            
            coords_4d = flowNet["model_in"]
            coords_3d = flowNet["model_out"]

            shape_model = self.shapeNet(coords_3d, preserve_grad=True)
            coords_sdf = shape_model['model_in']
            space_sdf = shape_model['model_out']
            grad_shape = gradient(space_sdf, coords_sdf).clone().detach()
            
            jacobian_flow = jacobian(coords_3d, coords_4d)[0]

            # grad_composed = grad_shape * jacobian_flow
            grad_shape = grad_shape.squeeze(0).unsqueeze(1)
            jacobian_flow = jacobian_flow.squeeze(0)
            grad = torch.bmm(grad_shape, jacobian_flow).squeeze(1).unsqueeze(0)

            # PDE constraints
            transport_constraint = transport_equation(grad)
            
            #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
            grad = grad[:,:,0:3] 
            
            pred_sdf = space_sdf
            # Initial-boundary constraints of the Eikonal equation at t=0
            sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
            normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
            sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
            grad_constraint = eikonal_at_time_constraint(grad, gt_sdf)

            return {
                "sdf_on_surface_constraint": sdf_on_surface_constraint.mean() * 3e3,
                "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 5e1,
                "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
                "grad_constraint": grad_constraint.mean() * 1e2,
                "transport_constraint": transport_constraint.mean()*1e2,
            }
        
    # def hybrid_transport(self, flowNet, gt):
    #         gt_sdf = gt["sdf"]
    #         gt_normals = gt["normals"]
            
    #         coords_4d = flowNet["model_in"]
    #         coords_3d = flowNet["model_out"]

    #         shape_model = self.shapeNet(coords_3d, preserve_grad=True)
    #         coords_sdf = shape_model['model_in']
    #         space_sdf = shape_model['model_out']
    #         grad_shape = gradient(space_sdf, coords_sdf).clone().detach()
            
    #         jacobian_flow = jacobian(coords_3d, coords_4d)[0]

    #         # grad_composed = grad_shape * jacobian_flow
    #         grad_shape = grad_shape.squeeze(0).unsqueeze(1)
    #         jacobian_flow = jacobian_flow.squeeze(0)
    #         grad = torch.bmm(grad_shape, jacobian_flow).squeeze(1).unsqueeze(0)

    #         # PDE constraints
    #         transport_constraint = transport_equation(grad)
            
    #         #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    #         grad = grad[:,:,0:3] 
            
    #         pred_sdf = space_sdf
    #         # Initial-boundary constraints of the Eikonal equation at t=0
    #         sdf_on_surface_constraint = on_surface_sdf_constraint(gt_sdf, pred_sdf)
    #         normal_on_surface_constraint = on_surface_normal_constraint(gt_sdf, gt_normals, grad) 
    #         sdf_off_surface_constraint = off_surface_sdf_constraint(gt_sdf, pred_sdf)
    #         grad_constraint = eikonal_at_time_constraint(grad, gt_sdf)


    #         v = torch.ones_like(gt_normals)
    #         tv = v*coords_4d[...,3].unsqueeze(-1)
    #         translations = coords_3d - (coords_4d[...,0:3] + tv)
    #         initial_constraint =  (translations.norm(dim=-1))**2
            
    #         dt = jacobian_flow[...,3].unsqueeze(0)
    #         derivative_constraint = ((dt-v).norm(dim=-1))**2

    #         return {
    #             "initial_constraint": initial_constraint.mean() * 3e3,
    #             "sdf_off_surface_constraint": sdf_off_surface_constraint.mean() * 5e1,
    #             "normal_on_surface_constraint": normal_on_surface_constraint.mean() * 1e2,
    #             "grad_constraint": grad_constraint.mean() * 1e2,
    #             "transport_constraint": transport_constraint.mean()*1e2,
    #         }