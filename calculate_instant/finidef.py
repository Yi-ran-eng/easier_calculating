import numpy as np
from typing import Literal,Callable
from scipy.linalg import lu_factor,lu_solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class FINIDF:
    def __init__(self,choice:Literal['thermalds'],need3d:bool):
        self.co=choice
        self._3d=need3d
    def __enter__(self):
        if self.co == 'thermalds':
            return self.thermal
    def thermal(self,duration_x:float,duration_t:float,N_x:int,N_t:int,alpha:float,x0:np.ndarray,
                T0:float,Tend:float,config1=None,config2=None):
        '''
        params durarion_x:空间范围，起始点默认是0
        params duration_t:时间模拟范围，初始时刻默认是0
        N_x,N_t:空间、时间网格点数
        alpha:热传导系数
        '''
        h=duration_x/N_x#空间步长
        k=duration_t/N_t#时间步长
        r=2*alpha**2*(k/h**2)
        #现在初始化温度分布,注意网格数和总点数的关系是一列/行总点数-1=网格数
        u=np.zeros((N_t+1,N_x+1),dtype=np.float32)
        #设定初值
        u[0]=x0
        u[:,0],u[:,-1]=np.full(u.shape[0],T0),np.full(u.shape[0],Tend)
        #构建宽对角系数矩阵,A的形状应该是只有内部点的方阵（总点数-2）
        A=np.diag([1+r for _ in range(u.shape[1]-2)])+np.diag([-r/2 for _ in range(u.shape[1]-3)],k=1)+\
        np.diag([-r/2 for _ in range(u.shape[1]-3)],k=-1)
        L,Piv=lu_factor(A)
        #现在构造b
        for i in range(N_t):
            U=u.copy()
            b=U[i,1:-1]#拿出当前时刻的b
            #更新b的首项和末项来确保正确计算下一个时刻的b
            b[0]=b[0]+r*U[i+1,0]#下一个时刻对应i+1,边界条件对应x索引为0
            b[-1]=b[-1]+r*U[i+1,-1]
            #然后解下一个时刻的b
            nxtb=lu_solve((L,Piv),b)
            u[i+1,1:-1]=nxtb#更新u
        #下面画图
        x=np.linspace(0,duration_x,N_x+1)
        t=np.linspace(0,duration_t,N_t+1)
        self.visiblil(x,t,u,self._3d,config1=config1,config2=config2)
    def visiblil(self,x1:np.ndarray,x2:np.ndarray,u:np.ndarray,create3d=False,**kw):
        '''
        params x1；横轴数据
        params x2:纵轴数据
        params u:值（value）
        '''
        X,Y=np.meshgrid(x1,x2)
        if X.size != u.size:
            raise ValueError('x1 or x2 cannt match u')
        congj=kw.get('config1')
        t=kw.get('title')
        if congj is not None:
            fig,ax=plt.subplots()
            im=ax.pcolormesh(X,Y,u,**congj)
        else:
            fig,ax=plt.subplots()
            im=ax.pcolormesh(X,Y,u)
        ax.set_title(t if t is not None else '')
        plt.colorbar(im,ax=ax)
        plt.show()
        if create3d:
            t2=kw.get('config2')
            fig=plt.figure()
            ax=fig.add_subplot(111,projection='3d')
            if t2 is not None:
                surf=ax.plot_surface(X,Y,u,**t2)
            else:
                surf=ax.plot_surface(X,Y,u)
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_zlabel('T')
            plt.show()
    def __exit__(self,x0,x1,x2):
        pass
def choiceparams(func:Callable):
    def wrapper(self,*args,**kw):
        '''
        输入时将前面几个当位置参数，后面当关键字参数
        以Euler为例，这里的args只包含N,x0,y0
        kw是两个关键字参数xarray和xdomain
        Euler里面用到的xmin和xmax在下面计算并传入，需要注意的是吧wrapper的传入参数和Euler实际传入参数分开看
        '''
        xarray=kw.get('xarray')
        xdomain=kw.get('xdomain')#不传自动为None
        N=args[0]
        if xarray is None and xdomain is not None:
            xmin,xmax=xdomain[0],xdomain[1]
            #说明可以直接算
            x=np.linspace(xmin,xmax,N)
        elif xarray is not None:
            xmin,xmax=xarray[0],xarray[-1]
            x=xarray
        return func(self,*args,xmin,xmax,x)
    return wrapper
class Eulerrungekuta:
    def __init__(self,func:callable):
        self.func=func
    def preparetwodera(self,x0,x):
        xsmall,xlarge=np.array([i for i in x if i <= x0]),np.array([i for i in x if i >= x0])
        return xsmall,xlarge
    @choiceparams
    def Euler(self,N,x0,y0,xmin,xmax,x):
        y=np.zeros(x.shape)
        h=(xmax-xmin)/(N-1)#注意区间数是N-1个
        if xmin == x0:
            y[0]=y0
            for i in range(N-1):
                y[i+1]=y[i]+h*self.func(x[i],y[i])
        elif xmax == x0:#反向欧拉
            y[-1]=y0
            h=-h
            for i in range(-1,-N-1,-1):
                y[i-1]=y[i]+h*self.func(x[i],y[i])
        elif xmin < x0 < xmax:#双向欧拉
            xsmall,xlarge=self.preparetwodera(x0,x)
            # ypre,yafter=np.zeros(xsmall.shape),np.zeros(xlarge.shape)
            # h1,h2=(x0-xsmall[0])/(len(xsmall)-1),(xlarge[-1]-x0)/(len(xlarge)-1)
            y1,y2=self.Euler(len(xsmall),x0,y0,xarray=xsmall),self.Euler(len(xsmall),x0,y0,xarray=xlarge)
            y1=y1[:-1]#去除最后一个，方便连接
            y=np.hstack((y1,y2))
        return y
    @choiceparams
    def advancedEu(self,N,x0,y0,xmin,xmax,x):
        y=np.zeros(x.shape)
        h=(xmax-xmin)/(N-1)
        if xmin == x0:
            y[0]=y0
            for j in range(N-1):
                y1=y[j]+h*self.func(x[j],y[j])
                y2=y[j]+h*self.func(x[j+1],y1)
                y[j+1]=(y1+y2)/2
        elif xmax == x0:
            y[-1]=y0
            h=-h
            for j in range(-1,-N-1,-1):
                y1=y[j]+h*self.func(x[j],y[j])
                y2=y[j]+h*self.func(x[j+1],y1)
                y[j-1]=(y1+y2)/2
        elif xmin < x0 < xmax:
            xsmall,xlarge=self.preparetwodera(x0,x)
            y1,y2=self.advancedEu(len(xsmall),x0,y0,xarray=xsmall),self.advancedEu(len(xsmall),x0,y0,xarray=xlarge)
            y1=y1[:-1]#去除最后一个，方便连接
            y=np.hstack((y1,y2))
        return y
    @choiceparams
    def rungekuta(self,N,x0,y0,xmin,xmax,x):
        y=np.zeros(x.shape)
        h=(xmax-xmin)/(N-1)
        if x0 == xmin:
            y[0]=y0
            for i in range(N-1):
                k1=self.func(x[i],y[i])
                k2=self.func(x[i]+h/2,y[i]+h*k1/2)
                k3=self.func(x[i]+h/2,y[i]+h*k2/2)
                k4=self.func(x[i]+h,y[i]+h*k3)
                y[i+1]=(k1+2*k2+2*k3+k4)*h/6+y[i]

        elif x0 == xmax:
            y[-1]=y0
            h=-h
            for i in range(-1,-N-1,-1):
                k1=self.func(x[i],y[i])
                k2=self.func(x[i]+h/2,y[i]+h*k1/2)
                k3=self.func(x[i]+h/2,y[i]+h*k2/2)
                k4=self.func(x[i]+h,y[i]+h*k3)
                y[i-1]=(k1+2*k2+2*k3+k4)*h/6+y[i]
        elif xmin < x < xmax:
            xsmall,xlarge=self.preparetwodera(x0,x)
            y1,y2=self.rungekuta(len(xsmall),x0,y0,xarray=xsmall),self.rungekuta(len(xsmall),x0,y0,xarray=xlarge)
            y1=y1[:-1]#去除最后一个，方便连接
            y=np.hstack((y1,y2))
        return y
# with FINIDF('thermalds',need3d=True) as the:
#     x=np.linspace(0,1,201)
#     y=(lambda x:40*np.sin(np.pi*x)+5)(x)
#     the(2,1,200,100,0.5,y,5,5,config1={'cmap':'viridis'})
def hanshu(x,y):
    return y-2*x/y
eg=Eulerrungekuta(hanshu)
y=eg.Euler(10,0,1,xdomain=(0,1))
print(y)