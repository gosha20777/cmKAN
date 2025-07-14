from cm_kan.ml.models import CmKAN, LightCmKAN
import torch


def test_cm_kan_create():
    """Test the creation of the CmKAN model"""
    model = CmKAN(
        in_dims=[3],
        out_dims=[3],
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
    )
    assert model is not None


def test_cm_kan_forward():
    """Test the forward pass of the CmKAN model"""
    model = CmKAN(
        in_dims=[3],
        out_dims=[3],
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
    )
    shape = (1, 3, 32, 32)
    x = torch.rand(shape)
    y = model(x)
    assert y.shape == shape


def test_light_cm_kan_create():
    """Test the creation of the LightCmKAN model"""
    model = LightCmKAN(
        in_dims=[3],
        out_dims=[3],
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
    )
    assert model is not None


def test_light_cm_kan_forward():
    """Test the forward pass of the LightCmKAN model"""
    model = LightCmKAN(
        in_dims=[3],
        out_dims=[3],
        grid_size=5,
        spline_order=3,
        residual_std=0.1,
        grid_range=[0.0, 1.0],
    )
    shape = (1, 3, 32, 32)
    x = torch.rand(shape)
    y = model(x)
    assert y.shape == shape
