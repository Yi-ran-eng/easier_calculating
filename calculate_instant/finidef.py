import numpy as np
from typing import Literal,Callable
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse import csr_matrix,linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod

class FINIDF:
    def __init__(self,choice:Literal['thermalds'],need3d:bool):
        self.co=choice
        self._3d=need3d
    def __enter__(self):
        if self.co == 'thermalds':
            return self.thermal
    @abstractmethod
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
    @abstractmethod
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

class Thermal_2d(FINIDF):
    def __init__(self,need3d:bool):
        super().__init__('thermalds',need3d)
        self.rowind=[]
        self.colind=[]
        self.dataval=[]
    def thermal(self,duration_x1:float,duration_t:float,duration_x2:float,N_x:int,N_t:int,alpha:float,x0:np.ndarray,
                T0:np.ndarray,Tend:list[float],config1=None,config2=None):
        '''
                  list2
              y___________
              |           |
        list1 |           |list3
              |           |
              _____________  x
                  list0
        '''
        h1,h2=duration_x1/N_x,duration_x2/N_x
        k=duration_t/N_t
        rx,ry=alpha*k/(h1**2),alpha*k/(h2**2)
        #initialize T0
        u=np.zeros((N_t,N_x,N_x))
        assert T0.ndim == 2,'initial t0 must be a 2d array'
        u[0]=T0
        u[:,0,:]=Tend[1]
        u[:,:,0]=Tend[0]
        u[:,-1,:]=Tend[3]
        u[:,:,-1]=Tend[2]
        Nint=N_x-2
        dotsym1=np.arange(1,N_x-1)
        dotsym2=np.arange(1,N_x-1)
        dot1,dot2=np.meshgrid(dotsym1,dotsym2)
        p=(dot1-1)+(dot2-1)*Nint
        for idx in range(Nint**2):
            i=idx%Nint+1
            j=idx//Nint+1
            self.rowind.append(idx)
            self.colind.append(idx)
            self.dataval.append(1+2*rx+2*ry)
            #left neighbor
            if i > 1:
                self.rowind.append(idx)
                self.colind.append(idx-1)
                self.dataval.append(-rx)
            if i < N_x -2:
                self.rowind.append(idx)
                self.colind.append(idx+1)
                self.dataval.append(-rx)
            #bottom
            if j > 1:
                self.rowind.append(idx)
                self.colind.append(idx-Nint)
                self.dataval.append(-ry)
            if j < N_x-2:
                self.rowind.append(idx)
                self.colind.append(idx+Nint)
                self.dataval.append(-ry)
        A=csr_matrix((self.dataval,(self.rowind,self.colind)),shape=(Nint**2,Nint**2))
        for k in range(N_t-1):
            #get the target layer
            utar=u[k,1:N_x-1,1:N_x-1]
            #put it as a 1d array
            utar_1d=utar.ravel(order='F').copy()
            for idx in range(Nint**2):
                i=idx % Nint + 1
                j=idx//Nint+1
                if i == 1:
                    utar_1d[idx]+=rx*u[k+1,0,j]
                if i == N_x-2:
                    utar_1d[idx]+=rx*u[k+1,N_x-1,j]
                if j == 1:
                    utar_1d[idx]+=ry*u[k+1,i,0]
                if j == N_x-2:
                    utar_1d[idx]+=ry*u[k+1,i,N_x-1]
            U_nxt_1d=linalg.spsolve(A,utar_1d)
            #back
            u[k+1,1:N_x-1,1:N_x-1]=U_nxt_1d.reshape(Nint,Nint)
        x=np.linspace(0,duration_x1,N_x)
        y=np.linspace(0,duration_x2,N_x)
        t=np.linspace(0,duration_t,N_t)
        self.visiblil(x,y,t,u,True,config1=config1,config2=config2)
    def visiblil(self, x1, x2,t, u, create3d=False, **kw):
        def _excplayer(pdict):
            if 'time_layers' not in pdict:
                congj=pdict
            else:
                congj=pdict.copy()
                congj.pop('time_layers')
            return congj
        N_t=len(t)
        timelayers=kw.get('time_layers',[0,N_t//2,N_t-1])
        nyplots=len(timelayers)
        congj=_excplayer(kw.get('config1'))
        t=kw.get('title','2d_visual')
        fig,axes=plt.subplots(nrows=1,ncols=nyplots)
        axes=[axes] if nyplots == 1 else axes
        X,Y=np.meshgrid(x1,x2)
        for idx,layer in enumerate(timelayers):
            if congj is not None:
                im=axes[idx].pcolormesh(X,Y,u[layer],**congj)
            else:
                im=axes[idx].pcolormesh(X,Y,u[layer])
            axes[idx].set_title(t)
            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('y')
            plt.colorbar(im,ax=axes[idx])
        print('???')
        plt.show()
        if create3d:
            #show the last time
            nt=N_t-1
            fig=plt.figure()
            ax=fig.add_subplot(projection='3d')
            conp=_excplayer(kw.get('config2'))
            surf=ax.plot_surface(X,Y,u[nt],**conp)
            ax.plot_surface(X,Y,u[nt])
            ax.set_title(t)
            fig.colorbar(surf,ax=ax,shrink=0.5,aspect=10)
        plt.show()
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

# def hanshu(x,y):
#     return y-2*x/y
# eg=Eulerrungekuta(hanshu)
# y=eg.Euler(10,0,1,xdomain=(0,1))
# print(y)

# 初始化
T0 = np.zeros((50, 50))
T0[20:30, 20:30] = 100  # 初始高温区域

# 边界条件 [左, 下, 右, 上]
Tend = [0.0, 0.0, 0.0, 0.0]

# 配置
config1 = {
    'cmap': 'hot',
    'time_layers': [0, 25, 49],  # 显示哪些时间层
}

config2 = {
    'cmap': 'coolwarm',
    'alpha': 0.9,
    'time_layers': [0, 25, 49]
}

with Thermal_2d(need3d=True) as solver:
    u = solver(
        duration_x1=1.0,    # x方向长度
        duration_t=1.0,     # 总时间
        duration_x2=1.0,    # y方向长度
        N_x=50,             # 网格数
        N_t=50,             # 时间步数
        alpha=0.01,         # 热扩散系数
        x0=T0,              # 初始场（你的参数名是x0，但代码里用T0）
        T0=T0,              # 初始温度
        Tend=Tend,          # 边界条件
        config1=config1,    # 2D可视化配置
        config2=config2     # 3D可视化配置
    )
