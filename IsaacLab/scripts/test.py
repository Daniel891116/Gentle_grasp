import torch

data_base = torch.load("baseline.pt", map_location="cpu")
data_adap = torch.load("adaptive.pt", map_location="cpu")

for name, d in [("baseline", data_base), ("adaptive", data_adap)]:
    print("=== ", name, " ===")
    print("ts_contact_force:", d["ts_contact_force"].shape,
          "mean=", d["ts_contact_force"].mean().item(),
          "max=", d["ts_contact_force"].max().item())
    print("ts_stiffness    :", d["ts_stiffness"].shape,
          "mean=", d["ts_stiffness"].mean().item(),
          "max=", d["ts_stiffness"].max().item())
    print("ts_slip_speed   :", d["ts_slip_speed"].shape,
          "mean=", d["ts_slip_speed"].mean().item(),
          "max=", d["ts_slip_speed"].max().item())
