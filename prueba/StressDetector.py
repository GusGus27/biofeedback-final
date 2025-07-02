import math

class StressDetector:
    def __init__(self, _interval, _samples_per_minute): #_sample_rate=240
        self.GSRValues = []  # Raw GSR values
        self.PAAValues = []
        self.SAXValues = []
        self.BaselineValues = []
        self.GSRmean = 0
        self.GSRsd = 0

        self.GSRxminute = _samples_per_minute  # Number of samples in one minute
        self.b_1 = -0.84
        self.b_2 = -0.25
        self.b_3 = 0.25
        self.b_4 = 0.84
        self.intervalSize = _interval
        self.e_cut = self.compute_e_cut()

    def has_stress(self):
        return True if self.process_stress() else False

    def process_stress(self):
        sax_preview = self.PAAValues[-1] if len(self.PAAValues) == 1 else self.PAAValues[-2]
        current_sax = self.PAAValues[-1]
        return current_sax - sax_preview > self.e_cut

    def process_gsr(self, gsr_values, baseline=[]):
        # Clear temporal vectors (PAA and SAX) for update time series
        self.PAAValues.clear()
        self.SAXValues.clear()
        self.GSRValues = baseline + gsr_values

        print(f"GSR Vector size: {len(self.GSRValues)}")

        # Filtering GSR values
        filtered_gsr = []
        print("Start Filtering")
        for i in range(len(self.GSRValues)):
            tm = 0
            if i < 50:
                for j in range(50 + i):
                    tm += self.GSRValues[j]
                tm /= (50 + i)
            elif i >= len(self.GSRValues) - 50:
                hh = 0
                for j in range(i - 50, len(self.GSRValues)):
                    tm += self.GSRValues[j]
                    hh += 1
                tm /= hh
            else:
                for j in range(100):
                    tm += self.GSRValues[i - 50 + j]
                tm /= 100
            filtered_gsr.append(tm)
        print("End Filtering", len(filtered_gsr))

        # Aggregating
        # Max value of each minute
        aggregated_gsr = []
        for i in range(len(filtered_gsr) // self.GSRxminute):
            max_val = max(filtered_gsr[i * self.GSRxminute: (i + 1) * self.GSRxminute])
            aggregated_gsr.append(max_val)
        print("End Aggregated", len(aggregated_gsr))

        # Z-normalization
        self.computeMeanSD(aggregated_gsr)
        if self.GSRsd == 0:          # ← NUEVO
            self.GSRsd = 1e-6        #   evita división por cero
        print(f"ComputeMeanSD: {self.GSRmean} - {self.GSRsd}")

        z_normalization = [(val - self.GSRmean) / self.GSRsd for val in aggregated_gsr]
        print("End Z", len(z_normalization))

        n = len(z_normalization)
        w = n // 3
        for i in range(1, w + 1):
            c_i = 0
            j_i = (n // w) * (i - 1) + 1
            j_f = (n // w) * i
            for j in range(j_i, j_f + 1):
                c_i += z_normalization[j - 1]
            c_i = (c_i * w) / n
            self.PAAValues.append(c_i)
            saxtmp = self.SAXvalue(c_i)
            self.SAXValues.append(saxtmp)
            print(f"PAA: {c_i} SAX: {saxtmp}")

    def SAXvalue(self, value):
        if value < self.b_1:
            return 1
        elif value < self.b_2:
            return 2
        elif value < self.b_3:
            return 3
        elif value < self.b_4:
            return 4
        else:
            return 5

    def computeMeanSD(self, aggregatedGSR):
        print("aggregatedGSR:", aggregatedGSR)
        self.GSRmean = sum(aggregatedGSR) / len(aggregatedGSR)
        self.GSRsd = math.sqrt(sum([(x - self.GSRmean) ** 2 for x in aggregatedGSR]) / len(aggregatedGSR))
        print(f"Mean: {self.GSRmean}, SD: {self.GSRsd}")

    def compute_e_cut(self):
        d = 1
        k = self.intervalSize * 2
        k_1 = self.intervalSize
        k_2 = self.intervalSize
        a_adwin = 1 / ((1 / k_1) + (1 / k_2))
        return math.sqrt((1 / (2 * a_adwin)) * math.log((k * 4) / d))

    def get_interval_size(self):
        return self.intervalSize

    def set_interval_size(self, interval):
        self.intervalSize = interval

    def get_sax_values(self):
        return self.SAXValues
