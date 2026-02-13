import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve,root,newton,brentq
from scipy.integrate import solve_ivp
import inspect
from functools import wraps
from typing import Literal,Callable
from finidef import FINIDF,Thermal_2d
def outerparams(*paramname):
    def deco(func):
        @wraps(func)
        def wrapper(*args,**kwar):
            #先获取默认值
            sig=inspect.signature(func)
            des={
                k:v.default for k,v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty
            }
            newkw=kwar.copy()
            for pam in paramname:
                if pam in newkw and newkw[pam] is None:
                    #这种情况就用默认值
                    newkw[pam]=des[pam]
            return func(*args,**newkw)
        return wrapper
    return deco
class _Baseicon:
    def setup(self,fig:plt.Figure,ax:Axes | Axes3D):
        self.fig=fig
        self.ax=ax
        self.dots=ax.scatter([],[],c='red',s=50,zorder=5) if not hasattr(ax,'get_zlim') else\
                                                    ax.scatter3D([],[],[],c='green',s=50,zorder=5)
        self.xpo,self.ypo=[],[]
        if hasattr(ax,'get_zlim'):
            self.zpo=[]
        self.fig.canvas.mpl_connect('button_press_event',lambda event:self.figclick(event))
    def figclick(self,event:MouseEvent):
        if event.inaxes != self.ax:
            return
        x,y=event.xdata,event.ydata
        self.xpo.append(x)
        self.ypo.append(y)
        if hasattr(self.ax,'get_zlim'):
            self.zpo=event
            self.dots._offsets3d=(self.xpo,self.ypo,self.zpo)
        self.dots.set_offsets(np.c_[self.xpo,self.ypo])
        plt.draw()
    def _clear(self):
        self.xpo,self.ypo=[],[]
        plt.draw()
class MixedCalcul(_Baseicon):
    @staticmethod
    @outerparams('y0')
    def solv(f,x_val,y0=0):
        y0array=np.full_like(x_val,y0,dtype=float)
        def eq(y,x):
            return f(x,y)
        sol,_,_,_ =fsolve(eq, y0array,args=(x_val,),full_output=True)
        return sol
    def bas_vivual(self,expr:sp.Basic,span:np.ndarray,targetvar:tuple,isfou=False,**kw):
        #转成函数
        assert isinstance(span,np.ndarray),'输入格式错误,span参数必须是np数组'
        adtitle='${'+sp.latex(expr)+'}$'
        if len(targetvar) == 1:
            f=sp.lambdify(targetvar,expr,'numpy')
            ys=f(span)
            fig,ax=plt.subplots()
            super().setup(fig,ax)
            ax.plot(span,ys,label='函数值')
            adtitle='${'+sp.latex(expr)+'}$'
            plt.title(adtitle)
            plt.show()
        
        elif len(targetvar) == 2:
            f=sp.lambdify(targetvar,expr,'numpy')
            if not isfou:
                x=span[0:1,:]
                y=span[1:2,:]
                X,Y=np.meshgrid(x,y)
                res=f(X,Y)
                fig=plt.figure()
                axx=fig.add_subplot(111,projection='3d')
                super().__init__(fig,axx)
                surf=axx.plot_surface(X,Y,res,cmap='viridis',edgecolor='none',alpha=0.8)
                fig.colorbar(surf,shrink=0.5,aspect=10,label='vars')
                axx.set_xlabel('X')
                axx.set_ylabel('Y')
                axx.set_zlabel('Z')
                plt.title(adtitle)
                plt.show()
            else:
                y0=kw.get('y0')
                xv=span.squeeze()
                solliner=MixedCalcul.solv(f,xv,y0=y0)
                plt.plot(span,solliner,c='green')
                plt.title(adtitle)
                plt.show()
    def _vision_order(self):
        latexp=[]
        if self.a != 0:
            latexp.append(f"{self.a:.2f}y''" if self.a != 1 else "y''")
        if self.b != 0:
            sign="+" if latexp and self.b > 0 else ''
            coef='' if abs(self.b) == 1 else f'{abs(self.b):.3f}'
            latexp.append(f"{sign}{coef}y'")
        if self.c != 0:
            sign='+' if latexp and self.c > 0 else ''
            latexp.append(f"{sign}{self.c:.3f}y")
        if self.d != 0:
            # sign='+' if latexp else ''
            if isinstance(self.d,sp.Basic):
                d_latex=sp.latex(self.d)
                sign=''
            else:
                d_latex=str(round(self.d,3))
                sign='+'
            latexp.append(f"{sign}{d_latex}")
            eqlate=''.join(latexp)+'=0'
            eqq=r"${"+eqlate+r"}$"
        return eqq
    def diff_visial(self,items:Literal['1','2'],orders:int,exper:list,targetvars,**kw):
        '''
        params order:微分方程的阶数，只支持1,2阶目前
        params exper:化成微分方程标准形式，ay''+by'+cy+d=0,d可能是含x的表达式，也可能是常数，这个列表存储[a,b,c,d]
        '''
        if items == '1':
            x,y=targetvars
            a,b,c,d=exper
            self.x,self.y=x,y
            self.a,self.b,self.c,self.d=a,b,c,d
            if isinstance(d,sp.Basic) or isinstance(d,sp.Symbol):
                d_np=sp.lambdify(targetvars,exper[-1],'numpy')
            else:
                d_np=d
            x_span=kw.get('span')#一维np数组
            if orders == 1:
                assert a == 0,'输入系数和阶数不匹配'
                def dy(x_dot,y_dot):
                    return np.array([(-c*y_dot-d_np(x_dot,y_dot))/b])
                y0=[kw.get('y0')] if kw.get('y0') is not None else [1]
                sols=solve_ivp(dy,x_span,y0,t_eval=np.linspace(0,10,1000))
                plt.plot(sols.t,sols.y[0])
                plt.xlabel('x')
                plt.ylabel('y')
                plt.show()
            else:
                '''
                注意齐次和非齐次分开写，如果是非齐次，那么等号右侧项是-d
                '''
                def d2y(x_dot,z):
                    y,ydot=z
                    F=-d_np(x_dot,ydot) if (not isinstance(d_np,float)) else -d_np
                    yddot=(F-c*y-b*ydot)/a
                    return [ydot,yddot]
                y0=kw.get('y0')
                x_span=kw.get('span')
                assert len(y0) == 2,'二阶微分方程需要提供两种微分方程'
                sols=solve_ivp(d2y,x_span,y0,t_eval=np.linspace(0,10,500))
                x=sols.t
                y,ydot=sols.y
                fig,axx=plt.subplots(ncols=2,nrows=1,figsize=(12,5))
                for i in range(2):
                    ax=axx[i]
                    if i == 0:
                        ax.plot(x,y,label='y(x)')
                    else:
                        ax.plot(x,ydot,label="y'(x)")
                    ax.legend()
                    ax.grid(True,alpha=0.7)
                tils=self._vision_order()
                fig.suptitle(tils,fontsize=14,y=0.98)
                plt.show()
class CertainDiff:
    def _1dthermal(self,x,Tinitial:np.ndarray | Callable,duration_x,duration_t,
                   Nx,Nt,alpha,x0,T0=0.0,Tend=0.0,config1=None,config2=None,**kw):
        '''
        crho *dTdt =alpha(d2Tdx2)
        '''
        inputs=locals()
        try:
            inputs.pop('kw')
        except KeyError:
            pass
        if isinstance(Tinitial,np.ndarray):
            with FINIDF('thermalds',need3d=True) as solver:
                solver.thermal(**inputs)
        else:
            initialparams=kw.get('initial')
            T0=Tinitial(x,**initialparams)
            newcoputs=inputs.copy()
            newcoputs['Tinitial']=T0
            with FINIDF('thermalds',need3d=True) as solver:
                solver.thermal(**inputs)
    def _2dthermal(self,duration_x1:float,duration_t:float,duration_x2:float,N_x:int,N_t:int,alpha:float,x0:np.ndarray,
                T0:np.ndarray,Tend:list[float],config1=None,config2=None):
        inputs=locals()
        with Thermal_2d(need3d=True) as solver:
            u=solver(
                **inputs  
            )
        return u
# x,y=sp.symbols('x y')
# m=MixedCalcul()
# m.bas_vivual(sp.exp(-x)+sp.sin(2*x),np.linspace(-3,3,100),(x,))
# m.bas_vivual(x-sp.cos(4*y+2),np.linspace(-1,1,1000),(x,y),isfou=True,y0=0)
# m.bas_vivual(x-sp.cos(4*y+2),np.vstack((np.linspace(-1,1,1000),np.linspace(-1,2,1000))),(x,y))
# m.diff_visial('1',2,[1,2*0.3*2*np.pi,4*np.pi**2,-10*sp.sin(2*x)],(x,y),span=[0,10],y0=[1,0])