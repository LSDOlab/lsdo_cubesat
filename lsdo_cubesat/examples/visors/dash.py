from lsdo_dash.base_dash import BaseDash


class Dash(BaseDash):

    def setup(self):
        # Define what variables to save

        self.set_clientID('simulator', write_stride=3)
        for varname in self.user_varnames:
            self.save_variable(varname, history=True)

        # Define frames
        self.add_frame(1,
                       height_in=8.,
                       width_in=12.,
                       nrows=2,
                       ncols=3,
                       wspace=0.4,
                       hspace=0.4)
