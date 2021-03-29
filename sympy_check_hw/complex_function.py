from sympy import symbols, diff, simplify, I, re, im

__all__ = ["MyComplexFxn"]

class MyComplexFxn:

    def __init__(self, **kwargs):
        """
            Construction of the Object
            Mode1:
                Object takes in 2 real valued scalar function that based on the real and the imaginary parts.
                E.G:
                    MyComplexFxn(x=x, y=y, u=x, v=y)
                Parameters needed:
                    x, y, u, v
                    where x,y are sympy.symbols and u, v are sympy expression
            mode2:
                Object takes in one function that maps from complex domain to complex domain.
                E.G:
                    MycomplexFxn(z=z, expr=z**2)
                Parameters needed:
                    z, expr
                    Where z is a sympy symbols and expr is a sympy expressiown.
            mode3:
                Object takes in one function, and the function is expressing in conjugate form.

            :exception
                Bunch of errors will be thrown if the type of the variables are not right.
        :param kwargs:
            None
        """

        def uv_diff():
            self.u_x = diff(self.u, x)
            self.u_y = diff(self.u, y)
            self.v_x = diff(self.v, x)
            self.v_y = diff(self.v, y)

        # case real parts, Mode 1
        if "u" in kwargs and "v" in kwargs:
            x, y = kwargs["x"], kwargs["y"]
            u, v = kwargs["u"], kwargs["v"]
            assert x.is_real and y.is_real, "u(x, y) and v(x, y) should taking real values as inputs"
            self.u, self.v = u, v
            uv_diff()
            return
        # Case complex functiom, mode 2.
        elif "z" in kwargs:
            z = kwargs["z"]
            expr = kwargs["expr"]
            assert z.is_real is None and z.is_imaginary is None, "Z must be a complex number."
            self.z = expr
            x, y = symbols("x y", real=True)
            expr = expr.subs([(z, x + y*I)])
            self.u = re(expr); self.v = im(expr)
            uv_diff()
            return
        raise RuntimeError("You didn't put in anything. ")

    def is_cauchy_riemann(self):
        """
            Return a boolean to indicates if the Riemann cauchy condition is satisified.
        :return:
            true if it is, false if it isn't
        """
        return simplify(self.u_y + self.v_x) == 0 and simplify(self.u_x - self.v_y) == 0


    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def get_vars():
        """
            Conveniently get the variables you need to define the parts function and the complex function
        :return:
        """
        return symbols("x", real=True), symbols("y", real=True),  symbols("z")




def main():
    x, y, z = MyComplexFxn.get_vars()
    TheComplxFxn = MyComplexFxn(z=z, expr=z ** 2)
    print(TheComplxFxn.u)
    print(TheComplxFxn.is_cauchy_riemann())
    pass


if __name__ == "__main__":
    main()