import csv, random, os, math

def clamp(x, lo, hi):
    if x < lo: return lo
    if x > hi: return hi
    return x

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def make_row():
    # Generate plausible features
    fails_24h = random.randint(0, 6)
    fails_10m = random.randint(0, 3)
    account_age_years = random.random() * random.random() * 10.0

    ip_rep = clamp(0.5 + random.random() * 0.7, 0.0, 1.0)
    if random.random() < 0.10:
        ip_rep = random.random() * 0.4  # bad IP

    geo_km = random.random() * 100.0
    if random.random() < 0.05:
        geo_km = 500.0 + random.random() * 4500.0  # far jump

    new_device = 1.0 if random.random() < 0.18 else 0.0
    time_anom = random.random() * random.random()
    pwd_change_7d = 1.0 if random.random() < 0.06 else 0.0

    velocity_h = random.random() * 2.0
    if random.random() < 0.07:
        velocity_h = 2.0 + random.random() * 8.0

    mfa_enabled = 1.0 if random.random() < 0.65 else 0.0

    # Hidden "true" risk rule -> score z
    z = -2.2
    z += 0.45 * fails_24h + 1.00 * fails_10m
    z += -0.25 * account_age_years
    z += -3.00 * ip_rep + 0.0012 * geo_km
    z += 1.10 * new_device + 2.20 * time_anom
    z += 1.30 * pwd_change_7d + 0.60 * velocity_h
    z += -1.10 * mfa_enabled
    z += 2.20 * (new_device * (1.0 - ip_rep))  # interaction
    if geo_km > 800.0:
        z += 1.40 * time_anom
    if mfa_enabled == 1.0 and account_age_years > 2.0:
        z += -0.90
    z += random.gauss(0.0, 0.6)  # noise

    # score -> probability -> label
    p = sigmoid(z)
    label = 1 if random.random() < p else 0

    return [
        fails_24h, fails_10m, account_age_years, ip_rep, geo_km,
        new_device, time_anom, pwd_change_7d, velocity_h, mfa_enabled,
        label
    ]

def main():
    os.system("rm -rf data")
    os.mkdir("data")

    print("Generating train.csv")
    file = open("data/train.csv", "w", newline="")
    writer = csv.writer(file)
    
    # Write header
    writer.writerow([
        "fails_24h", "fails_10m", "account_age_years", "ip_rep", "geo_km",
        "new_device", "time_anom", "pwd_change_7d", "velocity_h",
        "mfa_enabled", "label"
        ])

    # Create training data
    tdata = []
    for i in range(2000):
        tdata.append(make_row())

    # Write data
    for i in range(2000):
        writer.writerow(tdata[i])
    file.close()

    print("Generating coeffs.csv")
    file = open("data/coeffs.csv", "w", newline="")
    writer = csv.writer(file)
    writer.writerow([
        "fails_24h", "fails_10m", "account_age_years", "ip_rep", "geo_km",
        "new_device", "time_anom", "pwd_change_7d", "velocity_h",
        "mfa_enabled", "label"
        ])

    mean = []
    for i in range(11):
        curr_mean = 0
        for j in range(2000):
            curr_mean += tdata[j][i]
        curr_mean /= 2000
        mean.append(curr_mean)
    writer.writerow(mean)

    std = []
    for i in range(11):
        curr_std = 0
        for j in range(2000):
            curr_std += (tdata[j][i] - mean[i]) ** 2
        curr_std /= 2000
        curr_std = math.sqrt(curr_std)
        std.append(curr_std)
    writer.writerow(std)

    file.close()

    print("Generating val.csv")
    file = open("data/val.csv", "w", newline="")
    writer = csv.writer(file)
    
    # Write header
    writer.writerow([
        "fails_24h", "fails_10m", "account_age_years", "ip_rep", "geo_km",
        "new_device", "time_anom", "pwd_change_7d", "velocity_h",
        "mfa_enabled", "label"
        ])
    
    # Write data
    for i in range(500):
        writer.writerow(make_row())
    file.close()

if __name__ == "__main__":
    main()
