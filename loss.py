# coding: utf-8

from math import pi
from numpy import identity
import torch
from torch.functional import F
from util import gradient, jacobian, vector_dot, direction_aligment_on_surf
from util import divergence
from model import SIREN

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
    theta = coords[:,:,3].unsqueeze(-1)
    y = coords[:,:, 1].unsqueeze(-1)
    z = coords[:,:, 2].unsqueeze(-1)

    ft = grad[:,:,3].unsqueeze(-1)
    fy = grad[:,:,1].unsqueeze(-1)
    fz = grad[:,:,2].unsqueeze(-1)

    ty = (-torch.sin(theta)*y - torch.cos(theta)*z)
    tz = ( torch.cos(theta)*y - torch.sin(theta)*z)

    return (ft + ty*fy + tz*fz)**2
 
def sin_equation(grad, coords):
    #we consider the animate the x-coord using function sine
    pi = 3.14159265359
    theta = 2*pi*coords[...,3]
   
    return torch.abs( grad[...,3] + pi*torch.cos(theta)*grad[...,1] + grad[...,2])
 

def twist_vector_field(x):
    X = x[...,0].unsqueeze(-1)
    Y = x[...,1].unsqueeze(-1)
    Z = x[...,2].unsqueeze(-1)
    
    vx = - 2. * Y * Z
    vy = torch.zeros_like(vx)
    vz =   2. * Y * X
    return torch.cat((vx,vy,vz),dim=-1)


def twist_parametric_constraint(x,y):
    V = twist_vector_field(y)

    jacobian_flow = jacobian(y, x)[0]
    grad_t = jacobian_flow[...,3]
    diff = grad_t - V
    return (diff.norm(dim=-1))**2


def level_set_equation(grad, x):
    ft = grad[:,:,3].unsqueeze(-1)
    grad = grad[...,0:3]
    V = twist_vector_field(x)
    dot = vector_dot(grad, V)
    
    return (ft + dot)**2

def mean_curvature_equation(grad, x, scale = 0.00000001):
    ft = grad[:,:,3].unsqueeze(-1) # Partial derivative of the composedNet f with respect to the time t
   
    grad = grad[:,:,0:3] # Gradient of the composedNet f with respect to the space (x,y,z)
    grad_norm = torch.norm(grad, dim=-1).unsqueeze(-1)
    unit_grad = grad/grad_norm
    div = divergence(unit_grad, x)

    return (ft - scale*grad_norm*div)**2
    #return torch.abs(ft - grad_norm*div)

def implict_function(x, scale = 1.0):
    #sphere
    X = x[...,0]
    Y = x[...,1]    
    Z = x[...,2]
    return scale*(X**2 + Y**2 + Z**2 - 0.6**2)


def morphing_to_implict_function(grad, x, sdf, scale = 1.0):
    ft = grad[...,3].unsqueeze(-1)
    grad_3d = grad[...,0:3]
    grad_norm = torch.norm(grad_3d, dim=-1).unsqueeze(-1)

    target_deformation = implict_function(x, scale).unsqueeze(-1) - sdf

    grad_target_function = x[...,0:3]/torch.norm(x[...,0:3], dim=-1).unsqueeze(-1)
    additional_weight = vector_dot(grad_3d, grad_target_function)

    #return (ft - weight_deformation*grad_norm)**2
    return (ft - additional_weight*target_deformation*grad_norm)**2

def deformation_velocity(grad_3d, x, sdf, scale = 1., alpha = 2, beta = 0.01):

    target_deformation = implict_function(x, scale).unsqueeze(-1) - sdf

    #grad_target_function = x[...,0:3]/torch.norm(x[...,0:3], dim=-1).unsqueeze(-1)
    #additional_weight = vector_dot(grad_3d, grad_target_function)
    #grad_norm = torch.norm(grad_3d, dim=-1).unsqueeze(-1)
    
    #computing the mean curvature
    #mean_curv = 0.5*divergence(grad_3d/grad_norm, x)
    #mean_curv = 0.5*divergence(grad_3d, x)

    result = alpha*target_deformation#*additional_weight# + beta*mean_curv
    #result =  beta*mean_curv
    return result

def morphing_to_implict_function_with_tension(grad, x, sdf):
    ft = grad[...,3].unsqueeze(-1)
    grad_3d = grad[...,0:3]
    grad_norm = torch.norm(grad_3d, dim=-1).unsqueeze(-1)

    velocity = deformation_velocity(grad_3d, x, sdf)

    #V = -velocity*(grad_3d/grad_norm)

    #dot = vector_dot(grad_3d, V)

    #return (ft - weight_deformation*grad_norm)**2
    #return (ft - (additional_weight*target_deformation + 0.1*mean_curv)*grad_norm)**2
    return (ft - velocity*grad_norm)**2
    #return (ft + dot)**2


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
def compute_composed_gradient(coords_4d, coords_3d, shapeNet):
    shape_model = shapeNet(coords_3d, preserve_grad = True) 
    space_sdf = shape_model['model_out']
    
    return gradient(space_sdf, coords_4d)


class loss_flow(torch.nn.Module):
    def __init__(self, shapeNet):
        super().__init__()
        # Define the model.
        self.shapeNet = shapeNet
        self.shapeNet.cuda()
        self.shapeNet.eval()

    def forward(self, flowNet, gt):
        #return self.transport(flowNet, gt)
        #return self.implicit_transport(flowNet, gt)
        #return self.hybrid_transport(flowNet, gt)
        #return self.hybrid_morph(flowNet, gt)
        #return self.hybrid_morph_faster(flowNet, gt)
        # return self.hybrid_mean_curvature(flowNet, gt)
        #return self.hybrid_level_set_equation(flowNet, gt)
        return self.twist_space(flowNet, gt)
    
    # def forward(self, flowNet, model_flowNet, gt):
    #     return self.identity_inverse_level_set(flowNet, model_flowNet, gt)
    
    def transport(self, flowNet, gt):
        coords_4d = flowNet["model_in"]
        coords_3d = flowNet["model_out"]
 
        jacobian_flow = jacobian(coords_3d, coords_4d)[0]

        # grad_composed = grad_shape * jacobian_flow
        jacobian_flow = jacobian_flow.squeeze(0)
        
        v = torch.ones_like(coords_4d[...,0:3])
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
            grad = compute_composed_gradient(coords_4d, coords_3d, self.shapeNet)

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

    def hybrid_transport(self, flowNet, gt):
            
            coords_4d = flowNet["model_in"]
            coords_3d = flowNet["model_out"]

            grad = compute_composed_gradient(coords_4d, coords_3d, self.shapeNet)

            # PDE constraints
            implicit_transport_constraint = transport_equation(grad)
            
            translations = coords_3d - coords_4d[...,0:3]#(coords_4d[...,0:3] + tv)
            vector_field_constraint = (translations.norm(dim=-1))**2
            
            identity_constraint = torch.where(coords_4d[...,3]==0, vector_field_constraint, torch.zeros_like(vector_field_constraint)).unsqueeze(-1)

            return {
                "identity_constraint": identity_constraint.mean() * 1e4,
                "transport_constraint": implicit_transport_constraint.mean()*1e2, 
            }   

    # this loss is working
    def hybrid_morph(self, flowNet, gt):
            coords_4d = flowNet["model_in"]
            coords_3d = flowNet["model_out"]
            grad = compute_composed_gradient(coords_4d, coords_3d, self.shapeNet)

            shape_model = self.shapeNet(coords_3d) 
            space_sdf = shape_model['model_out']
            
            # PDE constraints
            morph_constraint = morphing_to_implict_function(grad, coords_4d, space_sdf)
            morph_constraint = torch.where(torch.abs(space_sdf)<0.1, morph_constraint, torch.zeros_like(morph_constraint))

            jacobian_flow = jacobian(coords_3d, coords_4d)[0]
            grad_t = jacobian_flow[...,3]
            velocity_constraint = (grad_t.norm(dim=-1)-torch.abs(implict_function(coords_4d))).unsqueeze(-1)**2

            translations = coords_3d - coords_4d[...,0:3]#(coords_4d[...,0:3] + tv)
            vector_field_constraint = (translations.norm(dim=-1))**2
            
            identity_constraint = torch.where(coords_4d[...,3]==-1, vector_field_constraint, torch.zeros_like(vector_field_constraint)).unsqueeze(-1)

            return {
                "identity_constraint": identity_constraint.mean() * 1e5,
                "morph_constraint": morph_constraint.mean()*1e2, 
                "velocity_constraint": velocity_constraint.mean(), 
            }  


    def hybrid_morph_faster(self, flowNet, gt):
            coords_4d = flowNet["model_in"]
            coords_3d = flowNet["model_out"]
            grad = compute_composed_gradient(coords_4d, coords_3d, self.shapeNet)

            shape_model = self.shapeNet(coords_3d) 
            space_sdf = shape_model['model_out']
            
            # PDE constraints
            #morph_constraint = morphing_to_implict_function(grad, coords_4d, space_sdf, scale = 1)
            morph_constraint = morphing_to_implict_function_with_tension(grad, coords_4d, space_sdf)
            morph_constraint *= torch.exp(-(space_sdf**2)/1e-2)

            # vector field norm constraint
            #jacobian_flow = jacobian(coords_3d, coords_4d)[0]
            #grad_t = jacobian_flow[...,3]

            #velocity = deformation_velocity(grad[...,0:3], coords_4d, space_sdf)
            #velocity_constraint = (grad_t.norm(dim=-1).unsqueeze(-1)-torch.abs(velocity))**2
            #velocity_constraint *= torch.exp(-(space_sdf**2)/1e-1)

            # identity constraint
            diff = coords_3d - coords_4d[...,0:3]
            id_constraint = vector_dot(diff, diff)
            identity_constraint = torch.where(coords_4d[...,3].unsqueeze(-1)==0, id_constraint, torch.zeros_like(id_constraint))

            #boundary constraint
            max_norm = torch.maximum(torch.maximum(torch.abs(coords_4d[...,0]), torch.abs(coords_4d[...,1])), torch.abs(coords_4d[...,2]))
            boundary_constraint = torch.where(max_norm.unsqueeze(-1) > 0.95, id_constraint, torch.zeros_like(id_constraint))
            
            morph_constraint = torch.where(max_norm.unsqueeze(-1) < 1.1, morph_constraint, torch.zeros_like(morph_constraint))

            # alignment between normals and deformations
            #align_constraint = direction_aligment_on_surf(grad_t, velocity*grad[..., 0:3])

            return {
                "identity_constraint": identity_constraint.sum(),
                "morph_constraint": morph_constraint.mean() * 1e2, 
                "boundary_constraint": boundary_constraint.sum()*1e-1, 
                #"velocity_constraint": velocity_constraint.mean() * 1e2,
                #"align_constraint": align_constraint.mean() * 1e2,
            }  


    def hybrid_mean_curvature(self, flowNet, gt):
            coords_4d = flowNet["model_in"]
            coords_3d = flowNet["model_out"]

            grad = compute_composed_gradient(coords_4d, coords_3d, self.shapeNet)

            # PDE constraints
            mean_curvature_constraint = mean_curvature_equation(grad, coords_4d, scale = 0.01)
            
            # identity constraint
            translations = coords_3d - coords_4d[...,0:3]
            diff_constraint = (translations.norm(dim=-1))**2
            identity_constraint = torch.where(coords_4d[...,3]==0, diff_constraint, torch.zeros_like(diff_constraint)).unsqueeze(-1)

            return {
                "identity_constraint": identity_constraint.sum(),
                "mean_curvature_constraint": mean_curvature_constraint.sum()*1e-2,
            }


    def twist_space(self, flowNet, gt):
            
            coords_4d = flowNet["model_in"]
            coords_3d = flowNet["model_out"]

            #grad = compute_composed_gradient(coords_4d, coords_3d, self.shapeNet)

            # PDE constraints
            #level_set_constraint = level_set_equation(grad, coords_4d)

            # identity constraint
            translations = coords_3d - coords_4d[...,0:3]
            diff_constraint = (translations.norm(dim=-1))**2
            identity_constraint = torch.where(coords_4d[...,3]==0, diff_constraint, torch.zeros_like(diff_constraint)).unsqueeze(-1)

            diff_twist_constraint = twist_parametric_constraint(coords_4d,coords_3d)

            return {
                "identity_constraint": identity_constraint.sum(),
                #"level_set_constraint": level_set_constraint.sum(),
                "diff_twist_constraint": diff_twist_constraint.sum(),
            }


    def identity_inverse_level_set(self, flowNet, model_flowNet, gt):
            
            coords_4d = model_flowNet["model_in"]
            coords_3d = model_flowNet["model_out"]

            #grad = compute_composed_gradient(coords_4d, coords_3d, self.shapeNet)

            # PDE constraints
            #level_set_constraint = level_set_equation(grad, coords_4d)

            # identity constraint
            translations = coords_3d - coords_4d[...,0:3]
            diff_constraint = (translations.norm(dim=-1))**2
            identity_constraint = torch.where(coords_4d[...,3]==0, diff_constraint, torch.zeros_like(diff_constraint)).unsqueeze(-1)

            # inverse constraint
            coords = torch.cat((coords_3d, -coords_4d[..., 3].unsqueeze(-1)), dim=-1)
            inverse_model = flowNet(coords)
            diff_inv = coords_3d - inverse_model["model_out"]
            diff_inv_constraint = (diff_inv.norm(dim=-1))**2
            inverse_constraint = torch.where(coords_4d[...,3]!=0, diff_inv_constraint, torch.zeros_like(diff_inv_constraint)).unsqueeze(-1)

            # PDE constraint
            diff_twist_constraint = diff_twist(coords_4d,coords_3d)

            return {
                "identity_constraint": identity_constraint.sum(),
                "inverse_constraint": inverse_constraint.sum(),
                #"level_set_constraint": level_set_constraint.sum(),
                "diff_twist_constraint": diff_twist_constraint.sum(),
            }


class loss_flow_morph(torch.nn.Module):
    def __init__(self, shapeNet):
        super().__init__()
        # Define the model.
        self.shapeNet = shapeNet
        self.shapeNet.cuda()

        self.targetNet = SIREN(
            n_in_features=3,
            n_out_features=1,
            hidden_layer_config=[128,128,128],
            w0=30
        )
        self.targetNet.load_state_dict(torch.load('./shapeNets/smpl_5-2x128_w0-30.pth'), strict=False)
        self.targetNet.cuda()
        print(self.targetNet)


    def forward(self, flowNet, gt):
        return self.hybrid_morph(flowNet, gt)


    def morphing_to_targetNet(self, grad, x, sdf):
        ft = grad[...,3].unsqueeze(-1)
        grad_3d = grad[...,0:3]
        grad_norm = torch.norm(grad_3d, dim=-1).unsqueeze(-1)

        coords = torch.zeros_like(x[...,0:3])#.clone().detach()
        coords[...,1]-=0.2
        coords += x[...,0:3]

        target_function = self.targetNet(coords)
        weight_deformation = target_function['model_out'] - sdf
        weight_deformation -= sdf

        additional_weight = 1#vector_dot(grad_3d, x[...,0:3]/torch.norm(x[...,0:3], dim=-1).unsqueeze(-1))

        #return (ft - weight_deformation*grad_norm)**2
        return (ft - additional_weight*weight_deformation*grad_norm)**2


    def hybrid_morph(self, flowNet, gt):
        coords_4d = flowNet["model_in"]
        coords_3d = flowNet["model_out"]
        grad = compute_composed_gradient(coords_4d, coords_3d, self.shapeNet)

        shape_model = self.shapeNet(coords_3d) 
        space_sdf = shape_model['model_out']
        
        # PDE constraints
        morph_constraint = self.morphing_to_targetNet(grad, coords_4d[...,0:3], space_sdf)
        #max_sdf = torch.abs(space_sdf).max()
        #morph_constraint = morph_constraint*(max_sdf-space_sdf)**2
        
        # jacobian_flow = jacobian(coords_3d, coords_4d)[0]
        # grad_t = jacobian_flow[...,3]
        #coords = torch.zeros_like(coords_4d[...,0:3])#.clone().detach()
        #coords[...,1]-=0.2
        coords = coords_4d[...,0:3]

        target_function = self.targetNet(coords)['model_out'] 
        # dist = target_function - space_sdf
        # velocity_constraint = (grad_t.norm(dim=-1).unsqueeze(-1)-torch.abs(dist))**2
    
        # identity constraint
        translations = coords_3d - coords_4d[...,0:3]#(coords_4d[...,0:3] + tv)
        vector_field_constraint = (translations.norm(dim=-1))**2
        identity_constraint = torch.where(coords_4d[...,3]==0, vector_field_constraint, torch.zeros_like(vector_field_constraint)).unsqueeze(-1)


        # target constraint
        target_constraint = target_function - space_sdf
        target_constraint = torch.where(coords_4d[...,3].unsqueeze(-1)==1, target_constraint**2, torch.zeros_like(target_constraint))#.unsqueeze(-1)

        return {
            "identity_constraint": identity_constraint.sum() * 1e1,
            #"morph_constraint": morph_constraint.mean()*1e2, 
            #"velocity_constraint": velocity_constraint.mean(), 
            "target_constraint": target_constraint.sum(), 
        }  
