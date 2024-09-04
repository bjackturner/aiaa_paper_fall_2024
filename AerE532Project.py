import numpy as np
import matplotlib.pyplot as plt

# Shock Relations theta-beta-mach
def th_tbm(mach, wave_angle, gamma): 
    th = np.arctan((2*(mach*np.sin(wave_angle))**2-2)/(mach**2*(gamma+np.cos(2*wave_angle))+2)/np.tan(wave_angle))
    return abs(th)

# Calculates mach 2 as well as pressure, density, and temperature ratio from mach 1 for an oblique shock and normal shock
def calculate_mach_2(mach_1,gamma,wave_angle):

    # Normal shock
    if wave_angle == np.pi/2:
        mach_2 = np.sqrt((mach_1**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * mach_1**2 - 1))
        pressure_ratio = 1 + (2 * gamma / (gamma + 1)) * (mach_1**2 - 1)
        density_ratio = ((gamma + 1) * mach_1**2)/(2 + ((gamma - 1) * mach_1**2))
        temperature_ratio = ((2 * gamma * mach_1**2 - (gamma - 1)) * ((gamma - 1) * mach_1**2 + 2))/((gamma + 1)**2 * mach_1**2)
        return mach_2, pressure_ratio, density_ratio, temperature_ratio
    
    # Oblique shock
    elif wave_angle < np.pi/2:
        th = th_tbm(mach_1, wave_angle, gamma)
        Mn = mach_1 * np.sin(wave_angle)
        mach_2 = (np.sqrt((Mn**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * Mn**2 - 1)))/np.sin(wave_angle - th)
        pressure_ratio = 1 + (2 * gamma / (gamma + 1)) * (Mn**2 - 1)
        density_ratio = ((gamma + 1) * Mn**2)/(2 + ((gamma - 1) * Mn**2))
        temperature_ratio = ((2 * gamma * Mn**2 - (gamma - 1)) * ((gamma - 1) * Mn**2 + 2))/((gamma + 1)**2 * Mn**2)
        return mach_2, pressure_ratio, density_ratio, temperature_ratio

# Calculates the normal mach number for an oblique and normal shock (kinda)
def calculate_mach_norm_2(mach_1, gamma, wave_angle):

    # If normal shock
    if wave_angle == np.pi/2:
        Mn1 = mach_1
        mach_2 = np.sqrt((mach_1**2 + (2 /(gamma - 1))) / (((2 * gamma)/(gamma - 1)) * mach_1**2 - 1))

        return Mn1, mach_2
    
    # If oblique shock
    elif wave_angle < np.pi/2:
        th = th_tbm(mach_1, wave_angle, gamma)
        Mn1 = mach_1 * np.sin(wave_angle)
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

def calculateOmega(alpha, mach_1, string, theta, wave_angle):

    if string == 'fast':
        omega1 = (np.sin(theta + wave_angle) + 1/mach_1) * alpha/np.sin(wave_angle)

    elif string == 'slow':
        omega1 = (np.sin(theta + wave_angle) - 1/mach_1) * alpha/np.sin(wave_angle)

    elif string == 'damped':
        Mn1, Mn2 = calculate_mach_norm_2(mach_1, gamma, wave_angle)
        alphaX, alphaY = findAlphaComp(alpha, theta)
        v0 = np.cos(wave_angle)/np.sin(wave_angle)

        omega1 = alphaX * (Mn1**2 - 1)/Mn1**2 + v0 * alphaY

    else:
        omega1 = np.sin(theta + wave_angle) * alpha/np.sin(wave_angle)

    return omega1

def calulateThetaV2(alpha, Mn1,theta,wave_angle, ub):
    omega1 = calculateOmega(alpha, Mn1, 'vorticity', theta, wave_angle)
    omega2 = omega1/ub

    alphaQ = omega2 - (np.cos(wave_angle)/(np.sin(wave_angle)*ub) * alphaY)
    alphaP = alphaY

    return (np.arcsin(alphaP/np.sqrt(alphaQ**2 + alphaP**2)) + np.pi/2) + (sigma2 * np.arcsin(alphaP*Mn2/np.sqrt(alphaQ**2 + alphaP**2)) - np.pi/2)

gamma = 5/3 # Right
mach_1 = 8 # Right
beta_list = [90 * np.pi/180, 65.89 * np.pi/180, 35.09 * np.pi/180] # Right

sigma1 = 1
sigma2 = 1
numPoints = 100

alpha = 0.056 # Right

string1 = 'fast' # entropy # vorticity
string2 = 'fast'

Angle = 'lower'

for j, wave_angle in enumerate(beta_list):

    Mn1, Mn2 = calculate_mach_norm_2(mach_1, gamma, wave_angle) # Right
    mach_2, pressure_ratio, density_ratio, temperature_ratio = calculate_mach_2(mach_1, gamma, wave_angle) # Right
    Mn1, Mn2 = calculate_mach_norm_2(mach_1, gamma, wave_angle) 
    ub = np.sqrt(temperature_ratio) * Mn2/Mn1

    beta2 = th_tbm(mach_1,wave_angle, gamma) # Right

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

    if wave_angle == 90 * np.pi/180:
        Normal = np.zeros_like(THETA)

    elif wave_angle == 65.89 * np.pi/180:
        Strong = np.zeros_like(THETA)

    elif wave_angle == 35.09 * np.pi/180:
        Weak = np.zeros_like(THETA)

    for i, theta in enumerate(THETA):
        alphaX, alphaY = findAlphaComp(alpha, theta) # Right

        omega1 = calculateOmega(alpha, Mn1, string1, theta, wave_angle)
        omega2 = omega1/ub

        alphaQ = omega2 - (np.cos(wave_angle)/(np.sin(wave_angle)*ub) * alphaY)
        alphaP = alphaY

        theta2 = (np.arcsin(alphaP/np.sqrt(alphaQ**2 + alphaP**2)) + np.pi/2) + (sigma2 * np.arcsin(alphaP*Mn2/np.sqrt(alphaQ**2 + alphaP**2)) - np.pi/2)
        #print((alphaP / np.sin(theta2)) - (alpha * np.sin(theta)/np.sin(theta2)))
        # print(np.sqrt(alphaX**2 + alphaY**2),np.sqrt(alphaP**2 + alphaQ**2))

        thetaV2 = calulateThetaV2(alpha, Mn1,theta,wave_angle,ub)

        Ec2 = makeEc2(string2, theta, Mn2, ub, gamma, alphaQ, alphaP, omega1, thetaV2, theta2)
        B = makeB(ub, gamma)
        Ci1 = makeCc2(string1, Mn1, theta)

        #print (Ci1)

        #print(Ci1- np.linalg.inv(Ec2)[:,0])

        if wave_angle == 90 * np.pi/180:
            Normal[i] = ub*(np.dot(np.dot(Ec2,B),np.transpose(Ci1))[0])
            
        elif wave_angle == 65.89 * np.pi/180:
            Strong[i] = ub*(np.dot(np.dot(Ec2,B),np.transpose(Ci1))[0])

        elif wave_angle == 35.09 * np.pi/180:
            Weak[i] = ub*(np.dot(np.dot(Ec2,B),np.transpose(Ci1))[0])

    if wave_angle == 90 * np.pi/180:
        plt.figure(j + 1)
        plt.plot(THETA/np.pi*180, Normal)
        plt.xlabel('Theta (Degrees)')
        plt.ylabel('Transmission Coefficient')
        plt.grid()

    elif wave_angle == 65.89 * np.pi/180:
        plt.figure(j + 1)
        plt.plot(THETA/np.pi*180, Strong)
        plt.xlabel('Theta (Degrees)')
        plt.ylabel('Transmission Coefficient')
        plt.grid()

    elif wave_angle == 35.09 * np.pi/180:
        plt.figure(j + 1)
        plt.plot(THETA/np.pi*180, Weak)
        plt.xlabel('Theta (Degrees)')
        plt.ylabel('Transmission Coefficient')
        plt.grid()

plt.show()





