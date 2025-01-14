from torch import nn
from ..layers.mw_isp import DWTForward, RCAGroup, DWTInverse, seq


class MWRCAN(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
        ):
        super(MWRCAN, self).__init__()
        c1 = 64
        c2 = 128
        c3 = 128
        n_b = 20
        self.head = DWTForward()

        self.down1 = seq(
            nn.Conv2d(in_channels * 4, c1, 3, 1, 1),
            nn.PReLU(),
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b)
        )

        self.down2 = seq(
            DWTForward(),
            nn.Conv2d(c1 * 4, c2, 3, 1, 1),
            nn.PReLU(),
            RCAGroup(in_channels=c2, out_channels=c2, nb=n_b)
        )

        self.down3 = seq(
            DWTForward(),
            nn.Conv2d(c2 * 4, c3, 3, 1, 1),
            nn.PReLU()
        )

        self.middle = seq(
            RCAGroup(in_channels=c3, out_channels=c3, nb=n_b),
            RCAGroup(in_channels=c3, out_channels=c3, nb=n_b)
        )
        
        self.up1 = seq(
            nn.Conv2d(c3, c2 * 4, 3, 1, 1),
            nn.PReLU(),
            DWTInverse()
        )

        self.up2 = seq(
            RCAGroup(in_channels=c2, out_channels=c2, nb=n_b),
            nn.Conv2d(c2, c1 * 4, 3, 1, 1),
            nn.PReLU(),
            DWTInverse()
        )

        self.up3 = seq(
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b),
            nn.Conv2d(c1, out_channels*4, 3, 1, 1)
        )

        self.tail = seq(
            DWTInverse(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.PReLU(),
        )

    def forward(self, x, c=None):
        c0 = x
        c1 = self.head(c0)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        m = self.middle(c4)
        c5 = self.up1(m) + c3
        c6 = self.up2(c5) + c2
        c7 = self.up3(c6) + c1
        out = self.tail(c7)

        return out