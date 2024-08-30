import numpy as np
import matplotlib.pyplot as plt

# Shock Relations 
def th_tbm(m,b, gamma): #explicit form of the theta-beta-M relation
    th = np.arctan((2*(m*np.sin(b))**2-2)/(m**2*(gamma+np.cos(2*b))+2)/np.tan(b))
    return abs(th)

def calculateM2(M1,gamma,beta):
    if beta == np.pi/2:
        M2 = np.sqrt((M1**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * M1**2 - 1))
        PR = 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)
        denR = ((gamma + 1) * M1**2)/(2 + ((gamma - 1) * M1**2))
        TR = ((2 * gamma * M1**2 - (gamma - 1)) * ((gamma - 1) * M1**2 + 2))/((gamma + 1)**2 * M1**2)
        return M2, PR, denR, TR
    
    elif beta < np.pi/2:
        th = th_tbm(M1, beta, gamma)
        Mn = M1 * np.sin(beta)
        M2 = (np.sqrt((Mn**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * Mn**2 - 1)))/np.sin(beta - th)
        PR = 1 + (2 * gamma / (gamma + 1)) * (Mn**2 - 1)
        denR = ((gamma + 1) * Mn**2)/(2 + ((gamma - 1) * Mn**2))
        TR = ((2 * gamma * Mn**2 - (gamma - 1)) * ((gamma - 1) * Mn**2 + 2))/((gamma + 1)**2 * Mn**2)
        return M2, PR, denR, TR
    
def calculateMn2(M1, gamma, beta):
    if beta == np.pi/2:
        Mn1 = M1
        M2 = np.sqrt((M1**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * M1**2 - 1))

        return Mn1, M2
    
    elif beta < np.pi/2:
        th = th_tbm(M1, beta, gamma)
        Mn1 = M1 * np.sin(beta)
        Mn2 = (np.sqrt((Mn1**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * Mn1**2 - 1)))

        return Mn1, Mn2

def makeB(ub, gamma):
    B = np.array([[2*ub - ((gamma-1)/(gamma+1)), 2*ub - 2*((gamma-1)/(gamma+1)), 0, -((2*gamma)/(gamma + 1))],
                  [((gamma-1)/(gamma+1)) - ub, 2*((gamma-1)/(gamma+1)) - ub, 0, ((2*gamma)/(gamma + 1))],
                  [0, 0, 1, 0],
                  [2/(gamma+1), 4/(gamma+1), 0, -((gamma-1)/(gamma+1))]])
    
    return B

def makeCd(ub, gamma, alphaq, alphap, omega1):
    b1 = 2 * (ub - ((gamma-1)/(gamma+1))) * (ub*alphaq)/omega1
    b2 = (((gamma-3)/(gamma+1)) -ub) * (ub*alphaq)/omega1
    b3 = (1-ub) * (alphap/omega1)
    b4 = 4/(gamma+1) * ((ub*alphaq)/omega1)

    return b1, b2, b3, b4

def makeEc2(string, theta, Mn2, ub, gamma, alphaq, alphap, omega1, thetaV2, theta2):

    d1, d2, d3, d4 = makeCc2(string, Mn2, theta2)

    b1, b2, b3, b4 = makeCd(ub, gamma, alphaq, alphap, omega1)

    # thetaV2 = np.pi/2 - theta

    m = 1/(np.sin(thetaV2) * (b3*d4 - b4*d3)/d4 + np.cos(thetaV2) * (b2*d4 - b4*d2)/d4)

    Ec2 = np.array([[0, -np.cos(thetaV2) * b4/d4, -np.sin(thetaV2) * b4/d4, np.sin(thetaV2) * b3/d4 + np.cos(thetaV2) * b2/d4],
                    [1/m, np.cos(thetaV2) * (b4*d1 - b1*d4)/d4, np.sin(thetaV2) * (b4*d1 - b1*d4)/d4, np.sin(thetaV2) * (b1*d3 - b3*d1)/d4 + np.cos(thetaV2) * (b1*d2 - b2*d1)/d4],
                 [0, -(b3*d4 - b4*d3)/d4, (b2*d4 - b4*d2)/d4, (b3*d2 - b2*d3)/d4],
                    [0, np.cos(thetaV2), np.sin(thetaV2), -np.sin(thetaV2) * d3/d4 - np.cos(thetaV2) * d2/d4]])
    
    return m * Ec2

# Cc2 Function (Incident Wave)
def makeCc2(string, Mn, theta):

    if string == 'fast':
        d1 = Mn**2
        d2 = Mn * np.cos(theta)
        d3 = Mn * np.sin(theta)
        d4 = 1

        return d1, d2, d3, d4

    elif string == 'slow':
        d1 = Mn**2
        d2 = -Mn * np.cos(theta)
        d3 = -Mn * np.sin(theta)
        d4 = 1

        return d1, d2, d3, d4

    elif string == 'entropy':
        d1 = 1
        d2 = 0
        d3 = 0
        d4 = 0

        return d1, d2, d3, d4

    elif string == 'vorticity':
        d1 = 0
        d2 = -np.sin(theta)
        d3 = np.cos(theta)
        d4 = 0

        return d1, d2, d3, d4

    else:
        d1 = 0
        d2 = 0
        d3 = 0
        d4 = 0

    return np.transpose([[d1, d2, d3, d4]])

def findAlphaComp(alpha, theta):

    alphaX = alpha * np.cos(theta)
    alphaY = alpha * np.sin(theta)

    return alphaX, alphaY

def calculateOmega(alpha, M1, string, theta, beta):

    if string == 'fast':
        omega1 = (np.sin(theta + beta) + 1/M1) * alpha/np.sin(beta)

    elif string == 'slow':
        omega1 = (np.sin(theta + beta) - 1/M1) * alpha/np.sin(beta)

    elif string == 'damped':
        Mn1, Mn2 = calculateMn2(M1, gamma, beta)
        alphaX, alphaY = findAlphaComp(alpha, theta)
        v0 = np.cos(beta)/np.sin(beta)

        omega1 = alphaX * (Mn1**2 - 1)/Mn1**2 + v0 * alphaY

    else:
        omega1 = np.sin(theta + beta) * alpha/np.sin(beta)

    return omega1

def calulateThetaV2(alpha, Mn1,theta,beta, ub):
    omega1 = calculateOmega(alpha, Mn1, 'vorticity', theta, beta)
    omega2 = omega1/ub

    alphaQ = omega2 - (np.cos(beta)/(np.sin(beta)*ub) * alphaY)
    alphaP = alphaY

    return (np.arcsin(alphaP/np.sqrt(alphaQ**2 + alphaP**2)) + np.pi/2) + (sigma2 * np.arcsin(alphaP*Mn2/np.sqrt(alphaQ**2 + alphaP**2)) - np.pi/2)

gamma = 5/3 # Right
M1 = 8 # Right
BETA = [90 * np.pi/180,65.89 * np.pi/180,35.09 * np.pi/180] # Right

sigma1 = 1
sigma2 = 1
numPoints = 100

alpha = 0.056 # Right

string1 = 'entropy' # entropy # vorticity
string2 = 'fast'

Angle = 'lower'

for beta in BETA:

    i = 0

    Mn1, Mn2 = calculateMn2(M1, gamma, beta) # Right
    M2, PR, denR, TR = calculateM2(M1, gamma, beta) # Right
    Mn1, Mn2 = calculateMn2(M1, gamma, beta) 
    ub = np.sqrt(TR) * Mn2/Mn1

    beta2 = th_tbm(M1,beta, gamma) # Right

    s = np.sqrt((gamma+1)/2 * ub*(1-ub))
    thetaAc = np.arcsin(1/np.sqrt(1 + s**2))
    thetaBc = np.arcsin(1/(Mn1 * np.sqrt(1 + s**2)))

    critAngleLower = (thetaAc + sigma1*thetaBc)*180/np.pi
    critAngleUpper = (np.pi - thetaAc + sigma1*thetaBc)*180/np.pi

    print(critAngleUpper)

    if Angle == 'lower':
        THETA = np.linspace(-critAngleLower*np.pi/180,critAngleLower*np.pi/180,numPoints)

    elif Angle == 'upper':
        THETA = np.concatenate((np.linspace(-np.pi/2, -critAngleLower*np.pi/180, numPoints//2),np.linspace(critAngleLower*np.pi/180, np.pi/2, numPoints//2)))

    if beta == 90 * np.pi/180:
        Normal = np.zeros_like(THETA)

    elif beta == 65.89 * np.pi/180:
        Strong = np.zeros_like(THETA)

    elif beta == 35.09 * np.pi/180:
        Weak = np.zeros_like(THETA)

    for theta in THETA:
        alphaX, alphaY = findAlphaComp(alpha, theta) # Right

        omega1 = calculateOmega(alpha, Mn1, string1, theta, beta)
        omega2 = omega1/ub

        alphaQ = omega2 - (np.cos(beta)/(np.sin(beta)*ub) * alphaY)
        alphaP = alphaY

        theta2 = (np.arcsin(alphaP/np.sqrt(alphaQ**2 + alphaP**2)) + np.pi/2) + (sigma2 * np.arcsin(alphaP*Mn2/np.sqrt(alphaQ**2 + alphaP**2)) - np.pi/2)
        #print((alphaP / np.sin(theta2)) - (alpha * np.sin(theta)/np.sin(theta2)))
        # print(np.sqrt(alphaX**2 + alphaY**2),np.sqrt(alphaP**2 + alphaQ**2))

        thetaV2 = calulateThetaV2(alpha, Mn1,theta,beta,ub)

        Ec2 = makeEc2(string2, theta, Mn2, ub, gamma, alphaQ, alphaP, omega1, thetaV2, theta2)
        B = makeB(ub, gamma)
        Ci1 = makeCc2(string1, Mn1, theta)

        #print (Ci1)

        #print(Ci1- np.linalg.inv(Ec2)[:,0])

        if beta == 90 * np.pi/180:
            Normal[i] = ub*(np.dot(np.dot(Ec2,B),np.transpose(Ci1))[0])
            

        elif beta == 65.89 * np.pi/180:
            Strong[i] = ub*(np.dot(np.dot(Ec2,B),np.transpose(Ci1))[0])

        elif beta == 35.09 * np.pi/180:
            Weak[i] = ub*(np.dot(np.dot(Ec2,B),np.transpose(Ci1))[0])

        i += 1

    if beta == 90 * np.pi/180:
        plt.plot(THETA/np.pi*180, Normal)

    elif beta == 65.89 * np.pi/180:
        plt.plot(THETA/np.pi*180, Strong)

    elif beta == 35.09 * np.pi/180:
        plt.plot(THETA/np.pi*180, Weak)

plt.xlabel('Theta (Degrees)')
plt.ylabel('Transmission Coefficient')
plt.grid()
# plt.ylim([0,20])
# plt.xlim([-66,66])
plt.show()



