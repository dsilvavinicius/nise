# coding: utf-8

import torch
from torch.functional import F
from nise.diff_operators import (divergence, gradient, mean_curvature,
                                 vector_dot)


def on_surface_sdf_constraint(gt_sdf, pred_sdf):
    return torch.where(
           gt_sdf == 0,
           # torch.abs(pred_sdf),
           (pred_sdf)**2,
           torch.zeros_like(pred_sdf)
        )


def off_surface_sdf_constraint(gt_sdf, pred_sdf):
    """
    This function forces gt_sdf and pred_sdf to be equals
    """
    result = torch.where(gt_sdf != -1, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf))
    return torch.where(gt_sdf != 0, result, torch.zeros_like(pred_sdf))


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
    # return torch.abs(grad.norm(dim=-1) - 1)


def eikonal_at_time_constraint(grad, gt_sdf):
    """
    This function forces the space-gradient of the SIREN function to be
    unitary at the time t: Eikonal Equation

    grad = (fx,fy,fz) : the space-gradient of the SIREN function
    coords = (x,y,z,t) : spacetime point
    """
    eikonal = (grad.norm(dim=-1) - 1.)**2

    return torch.where(
       gt_sdf != -1,
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


def transport_equation(grad):
    """transport along with (0,1,1)
    # f_t + b.(f_x, f_y, f_z) = 0
    # grad (f) = (f_x, f_y, f_z, f_t)"""

    ft = grad[:, :, 3].unsqueeze(-1)
    fy = grad[:, :, 1].unsqueeze(-1)
    fz = grad[:, :, 2].unsqueeze(-1)

    return (ft + (fy + fz))**2


def mean_curvature_equation(grad, x, scale=0.00000001):
    ft = grad[..., 3].unsqueeze(-1)  # Partial derivative of the SIREN function f with respect to the time t

    grad = grad[..., 0:3]  # Gradient of the SIREN function f with respect to the space (x,y,z)
    grad_norm = torch.norm(grad, dim=-1).unsqueeze(-1)
    unit_grad = grad/grad_norm
    div = divergence(unit_grad, x)

    return (ft - scale*grad_norm*div)**2
    # return torch.abs(ft - grad_norm*div)


def morphing_to_sphere(grad, x):
    ft = grad[:,:,3]
    grad = grad[:,:,0:3]
    grad_norm = torch.norm(grad, dim=-1)

    # sphere
    dist = x[...,0]**2 + x[...,1]**2 + x[...,2]**2 - 0.5

    return torch.abs(ft - grad_norm*dist)


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


class LossMeanCurvature(torch.nn.Module):
    def __init__(self, scale=0.001):
        super(LossMeanCurvature, self).__init__()
        self.scale = scale

    def forward(self, X, gt):
        gt_sdf = gt["sdf"]
        gt_normals = gt["normals"]
        coords = X["model_in"]
        pred_sdf = X["model_out"]

        grad = gradient(pred_sdf, coords)

        # PDE constraints
        mean_curvature_constraint = mean_curvature_equation(grad, coords, scale=self.scale)
        # mean_curvature_constraint = mean_curvature_equation(grad, coords, scale=0.025) #for dumbbell

        #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
        grad = grad[..., 0:3]

        # grad_len = grad.norm(dim=-1).unsqueeze(-1)
        # mean_curvature_constraint = torch.where(grad_len<10. , mean_curvature_constraint, torch.zeros_like(mean_curvature_constraint))
        # mean_curvature_constraint = torch.where(grad_len>1e-3, mean_curvature_constraint, torch.zeros_like(mean_curvature_constraint))

        # Initial-boundary constraints of the Eikonal equation at t=0
        sdf_constraint = torch.where(
            gt_sdf != -1,
            (gt_sdf - pred_sdf) ** 2,
            torch.zeros_like(pred_sdf)
        )

        # Hack to calculate the normal constraint only on points whose normals
        # lie on the surface, since we mark all others with -1 in all coordinates.
        # Note that the valid normals must have unit length.
        normal_constraint = torch.where(
            gt_normals[..., 0].unsqueeze(-1) != -1.,
            1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
            torch.zeros_like(grad[..., :1])
        )

        return {
            "sdf_constraint": sdf_constraint.mean() * 2e3, # 1e3
            "normal_constraint": normal_constraint.mean() * 1e1,
            "mean_curvature_constraint": mean_curvature_constraint.mean() * 1e3,
        }


def loss_mean_curv_with_restrictions(X, gt):
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)

    # PDE constraints
    mean_curvature_constraint = mean_curvature_equation(grad, coords, scale=0.001)
    mean_curvature_constraint = torch.where( coords[..., 0].unsqueeze(-1)<0, grad[...,3].unsqueeze(-1)**2, mean_curvature_constraint)

    #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
    grad = grad[...,0:3]

    sdf_constraint = torch.where( gt_sdf!=-1, (gt_sdf - pred_sdf) ** 2, torch.zeros_like(pred_sdf))
    normal_constraint = torch.where(
           gt_sdf!=-1,
           1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
           torch.zeros_like(grad[..., :1])
    )

    return {
        "sdf_constraint": sdf_constraint.mean() * 1e3,
        "normal_constraint": normal_constraint.mean() * 1e1,
        "mean_curvature_constraint": mean_curvature_constraint.mean()*1e3,
    }


# Equation 1 of "Geometric processing with neural fields"
class loss_NFGP(torch.nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        # Define the model.
        self.model = trained_model
        self.model.cuda()

    def forward(self, X, gt):
        gt_sdf = gt["sdf"]
        gt_normals = gt["normals"]

        coords = X["model_in"]
        pred_sdf = X["model_out"]

        grad = gradient(pred_sdf, coords)

        # PDE constraints
        #mean_curvature_constraint = mean_curvature_equation(grad, coords, scale=0.001)
        K_g = mean_curvature(grad[...,0:3], coords)

        # trained model
        trained_model = self.model(coords[...,0:3])
        trained_model_out = trained_model['model_out']
        trained_model_in = trained_model['model_in']
        grad_trained_model = gradient(trained_model_out, trained_model_in)
        K_f = mean_curvature(grad_trained_model, trained_model_in)

        GPNF_constraint = (K_g - (coords[...,3].unsqueeze(-1)+1)*K_f)**2
        #GPNF_constraint = (K_g - K_f)**2

        tau = 50
        #V_tau = torch.maximum(torch.abs(K_g), torch.abs(K_f))
        V_tau = torch.abs(K_f)
        GPNF_constraint = torch.where( V_tau<tau, GPNF_constraint, torch.zeros_like(GPNF_constraint))
        GPNF_constraint = torch.where(torch.abs(trained_model_out)<0.1, GPNF_constraint, torch.zeros_like(GPNF_constraint))

        #restricting the gradient (fx,ty,fz, ft) of the SIREN function f to the space: (fx,ty,fz)
        grad = grad[...,0:3]

        # Initial-boundary constraints of the Eikonal equation at t=0
        sdf_constraint = torch.where( gt_sdf!=-1, (trained_model_out - pred_sdf) ** 2, torch.zeros_like(pred_sdf))

        sdf_off_constraint = torch.where( V_tau>tau, (trained_model_out - pred_sdf) ** 2, torch.zeros_like(pred_sdf))
        sdf_off_constraint = torch.where(gt_sdf==-1, sdf_off_constraint, torch.zeros_like(pred_sdf))

        normal_constraint = torch.where(
            gt_sdf!=-1,
            1 - F.cosine_similarity(grad, grad_trained_model, dim=-1)[..., None],
            torch.zeros_like(grad[..., :1])
        )

        return {
            "sdf_constraint": sdf_constraint.mean() * 1e4,
            "sdf_off_constraint": sdf_off_constraint.mean() * 1e4,
            "normal_constraint": normal_constraint.mean() * 1e2,
            "GPNF_constraint": GPNF_constraint.mean()*1e-1,
            "grad_constraint": eikonal_constraint(grad).mean() * 1e2,
        }


class LossVectorField(torch.nn.Module):
    def __init__(self, trained_model, centers, spreads):
        super().__init__()
        # Define the model.
        self.model = trained_model
        self.model.cuda()

        self.centers = centers
        self.spreads = spreads 

    def source_vector_field(self,x, center = [0,0,0], spreads = [5,5,5] ):
        X = x[...,0].unsqueeze(-1)
        Y = x[...,1].unsqueeze(-1)
        Z = x[...,2].unsqueeze(-1)

        vx = X-center[0]
        vy = Y-center[1]
        vz = Z-center[2]

        gaussian = torch.exp(-(vx**2/(2*spreads[0]**2)+vy**2/(2*spreads[1]**2)+vz**2/(2*spreads[2]**2)))

        return gaussian*torch.cat((vx,vy,vz),dim=-1)


    def level_set_equation(self, grad, x):
        
        # time derivative of the network
        ft = grad[:, -1].unsqueeze(-1)

        # defining the vector fields using gaussians        
        V = torch.zeros_like(x[..., :-1])
        for center, spread in zip(self.centers, self.spreads):
            V += self.source_vector_field(x, center, spread)

        # creating the corresponding level set equation
        dot = vector_dot(grad[...,:-1], V)

        return (ft + dot)**2

    def forward(self, X, gt):
        coords = X["model_in"]
        pred_sdf = X["model_out"]

        grad = gradient(pred_sdf, coords)

        # PDE constraints
        level_set_constraint = self.level_set_equation(grad, coords)

        # trained model
        trained_model = self.model(coords[...,0:3])
        trained_model_out = trained_model['model_out']
        trained_model_in = trained_model['model_in']
        grad_trained_model = gradient(trained_model_out, trained_model_in).detach()

        # Initial condition at t=0
        time = coords[...,3].unsqueeze(-1)
        sdf_constraint = torch.where( time == 0, (trained_model_out.detach() - pred_sdf) ** 2, torch.zeros_like(pred_sdf))

        normal_constraint = torch.where(
            time == 0,
            1 - F.cosine_similarity(grad[...,0:3], grad_trained_model, dim=-1)[..., None],
            torch.zeros_like(grad[..., :1])
        )

        return {
            "sdf_constraint": sdf_constraint.mean()*5e3,
            "normal_constraint": normal_constraint.mean()*5e2,
            "level_set_constraint": level_set_constraint.mean()*1e3,
        }


class LossMorphingNI(torch.nn.Module):
    """Morphing between two neural implict functions."""
    def __init__(self, trained_models, times):
        super().__init__()
        # Define the models
        self.model1 = trained_models[0]
        self.model2 = trained_models[1]
        self.t1 = times[0]
        self.t2 = times[1]

    def morphing_to_NI(self, grad, sdf, coords, sdf_target, grad_target, scale=1):
        ft = grad[..., 3].unsqueeze(-1)
        grad_3d = grad[..., :3]
        grad_norm = torch.norm(grad_3d, dim=-1).unsqueeze(-1)

        # unit_grad = grad/grad_norm
        # div = divergence(unit_grad, coords)

        target_deformation = sdf_target - sdf

        # additional_weight = vector_dot(grad_3d, grad_target)

        target_deformation *= torch.exp(-sdf**2)  # gives priority to the zero-level set

        # deformation = - scale*target_deformation - 0.0005*div
        # deformation = - 0.0005*div
        deformation = - scale * target_deformation

        return (ft + deformation * grad_norm) ** 2

    def vector_field(self, coords, grad_src, grad_dst, t_src, t_dst):
        grad_norm_src = torch.norm(grad_src, dim=-1).unsqueeze(-1)
        grad_norm_dst = torch.norm(grad_dst, dim=-1).unsqueeze(-1)
        V_src = grad_src/grad_norm_src
        V_dst = grad_dst/grad_norm_dst

        # return V_src

        time = coords[..., 3].unsqueeze(-1)

        len = t_dst - t_src
        time = (time-t_src)/len

        V = (1-time)*V_src + time*V_dst
        return V

    def level_set_equation(self, grad, sdf, coords, sdf_target, grad_source, grad_target, scale=1):
        ft = grad[:, :, 3].unsqueeze(-1)

        target_deformation = sdf_target - sdf

        # additional_weight = vector_dot(grad_3d, grad_target)

        target_deformation *= torch.exp(-sdf**2) #gives priority to the zero-level set

        deformation = -scale * target_deformation

        V = deformation * self.vector_field(
            coords, grad_source, grad_target, self.t1, self.t2
        )

        dot = vector_dot(grad[..., :3], V)

        return (ft + dot)**2

    def loss_i4d_interpolation(self, X, gt):
        coords = X["model_in"]
        pred_sdf = X["model_out"]

        grad = gradient(pred_sdf, coords)

        # trained model1
        trained_model1 = self.model1(coords[..., 0:3])
        trained_model1_out = trained_model1['model_out']
        trained_model1_in = trained_model1['model_in']
        grad_trained_model1 = gradient(trained_model1_out, trained_model1_in)

        # trained model2
        trained_model2 = self.model2(coords[..., 0:3])
        trained_model2_out = trained_model2['model_out']
        trained_model2_in = trained_model2['model_in']
        grad_trained_model2 = gradient(trained_model2_out, trained_model2_in)

        morphing_constraint = self.morphing_to_NI(
            grad, pred_sdf, coords, trained_model2_out, grad_trained_model2,
            scale=0.5
        )
        # morphing_constraint = self.level_set_equation(grad, pred_sdf, coords, trained_model2_out, grad_trained_model1, grad_trained_model2, scale=10)

        # Restricting the gradient (fx,ty,fz, ft) of the SIREN function f to
        # the space: (fx,ty,fz)
        grad = grad[..., 0:3]
        time = coords[..., 3].unsqueeze(-1)

        # Initial-boundary constraints of the Eikonal equation at t=0
        sdf_constraint = torch.where(
            time == self.t1,
            (trained_model1_out - pred_sdf) ** 2,
            torch.zeros_like(pred_sdf)
        )
        sdf_constraint = torch.where(
            time == self.t2,
            (trained_model2_out - pred_sdf) ** 2,
            sdf_constraint
        )

        normal_constraint = torch.where(
            time == self.t1,
            1 - F.cosine_similarity(grad, grad_trained_model1, dim=-1)[..., None],
            torch.zeros_like(grad[..., :1])
        )

        normal_constraint = torch.where(
            time == self.t2,
            (1 - F.cosine_similarity(grad, grad_trained_model2, dim=-1)[..., None]),
            normal_constraint
        )

        return {
            "sdf_constraint": sdf_constraint.mean() * 1e4,
            "normal_constraint": normal_constraint.mean() * 1e1,
            # "morphing_constraint": morphing_constraint.mean() * 1e3,
            "morphing_constraint": morphing_constraint.mean() * 1e1,
        }

    def loss_lipschitz_interpolation(self, X, gt):
        coords = X["model_in"]
        pred_sdf = X["model_out"]

        # trained model1
        trained_model1 = self.model1(coords[..., :3])
        trained_model1_out = trained_model1['model_out'].detach()

        # trained model2
        trained_model2 = self.model2(coords[..., :3])
        trained_model2_out = trained_model2['model_out'].detach()

        # Initial-boundary constraints of the Eikonal equation at t=0
        sdf_constraint = torch.where(
            coords[..., 3].unsqueeze(-1) == self.t1,
            (trained_model1_out - pred_sdf) ** 2,
            torch.zeros_like(pred_sdf)
        )
        sdf_constraint = torch.where(
            coords[..., 3].unsqueeze(-1) == self.t2,
            (trained_model2_out - pred_sdf) ** 2,
            sdf_constraint
        )

        return {
            "sdf_constraint": sdf_constraint.mean()*1e2,
        }

    def forward(self, X, gt):
        return self.loss_i4d_interpolation(X, gt)
        # return self.loss_lipschitz_interpolation(X, gt)


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
    const_constraint = torch.abs(grad[:,:,3])
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
        "const_constraint": const_constraint.mean() * 1e2,
        "grad_constraint_spacetime": grad_constraint_spacetime.mean() * 1e2,
    }
