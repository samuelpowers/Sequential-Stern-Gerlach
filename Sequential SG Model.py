import scipy.special
import math
from math import factorial as mf
import numpy as np
import matplotlib.pyplot as plt



# This program generates probability distributions over theta_ab for any choice of n, j, m_a1, m_b2, and possibly l_a1.
# It is not written for computational speed!

# Begin user Inputs:
# Note: computational time increases with increasing n, increasing j, and decreasing magnitude of m and l.
n = 200
j = 2
m_a1 = 0
m_b2 = 0

# Treat l_at as a nuisance variable (Increases computational time)?
s_l_a1 = False
# If false, l_a1 will be taken as an additional conditioning variable, which decreases computational time significantly.
l_a1 = 60        # If s_l_a1 = False, l_a1 must be chosen. This value must not exceed |(n/2)-j| in magnitude and |l_a1 - |(n/2)-j|| must be an integer.
# End user inputs


# Handle errors in choosing l_a1
error = False
# Check l_a1
if not s_l_a1:
    if not abs(l_a1 - n/2 - j)%1==0 or abs(l_a1) > abs(n/2 - j):
        error = True
        print('Choice of l_a1 is invalid!')


# Define quantum numbers in terms of counts:
# Columns: [AA,AB,BA,BB,CC,CD,DC,DD]
n_def = [1, 1, 1, 1, 1, 1, 1, 1]
j_def = [0, 0, 0, 0, 1/2, 1/2, 1/2, 1/2]
m_a1_def = [0, 0, 0, 0, 1/2, 1/2, -1/2, -1/2]
m_b2_def = [0, 0, 0, 0, 1/2, -1/2, 1/2, -1/2]
l_a1_def = [1/2, 1/2, -1/2, -1/2, 0, 0, 0, 0]
l_b2_def = [1/2, -1/2, 1/2, -1/2, 0, 0, 0, 0]
b_map_def = [0, 1, 1, 0, 0, 1, 1, 0]
DD_def = [0, 0, 0, 0, 0, 0, 0, 1]

# We use DD rather than the path quantum number nu in this program because its range is simpler to define.
# The quantum number nu will be calculated before calculating Upsilon_a and Upsilon_b.
# With all other quantum numbers fixed, each unique choice of DD is equivalent to a unique choice of nu.

# Invert the quantum number matrix, which will be used to map from quantum numbers to counts
coefficient_array = np.linalg.inv(np.array([n_def, j_def, m_a1_def, m_b2_def, l_a1_def, l_b2_def, b_map_def, DD_def]))

# Calculate the cardinality of the ontic states in Alice's and Bob's contextual sets (\varepsilon).
def card_alice(counts_):
    # Count index: AA=0, AB=1, BA=2, BB=3, CC=4, CD=5, DC=6, DD=7

    # x = True bypasses numerical approximations used in scipy.special.factorial. This increases computational time.
    # x= False leads to numerical instabilities and overflow issues for large n (no longer ints)
    x = True

    # The base-4 counts associated with Alice's event
    A_a1 = counts_[0] + counts_[1]
    B_a1 = counts_[2] + counts_[3]
    C_a1 = counts_[4] + counts_[5]
    D_a1 = counts_[6] + counts_[7]

    # The factorials we will need
    n_a = scipy.special.factorial(A_a1, exact=x)
    n_b = scipy.special.factorial(B_a1, exact=x)
    n_c = scipy.special.factorial(C_a1, exact=x)
    n_d = scipy.special.factorial(D_a1, exact=x)
    d_aa = scipy.special.factorial(counts_[0], exact=x)
    d_ab = scipy.special.factorial(counts_[1], exact=x)
    d_ba = scipy.special.factorial(counts_[2], exact=x)
    d_bb = scipy.special.factorial(counts_[3], exact=x)
    d_cc = scipy.special.factorial(counts_[4], exact=x)
    d_cd = scipy.special.factorial(counts_[5], exact=x)
    d_dc = scipy.special.factorial(counts_[6], exact=x)
    d_dd = scipy.special.factorial(counts_[7], exact=x)

    # Calculating the cardinality of Alice's contextual set
    cardinality_alice = (n_a*n_b*n_c*n_d/(d_aa*d_ab*d_ba*d_bb*d_cc*d_cd*d_dc*d_dd))

    return cardinality_alice


def card_bob(counts_):
    # Count index: AA=0, AB=1, BA=2, BB=3, CC=4, CD=5, DC=6, DD=7

    # x = True bypasses numerical approximations used in scipy.special.factorial.
    # x= False leads to numerical instabilities and overflow issues for large n (no longer ints)
    x = True

    # The base-4 counts associated with Bob's event
    A_b2 = counts_[0] + counts_[2]
    B_b2 = counts_[1] + counts_[3]
    C_b2 = counts_[4] + counts_[6]
    D_b2 = counts_[5] + counts_[7]

    # The factorial we will need
    n_a = scipy.special.factorial(A_b2, exact=x)
    n_b = scipy.special.factorial(B_b2, exact=x)
    n_c = scipy.special.factorial(C_b2, exact=x)
    n_d = scipy.special.factorial(D_b2, exact=x)
    d_aa = scipy.special.factorial(counts_[0], exact=x)
    d_ab = scipy.special.factorial(counts_[1], exact=x)
    d_ba = scipy.special.factorial(counts_[2], exact=x)
    d_bb = scipy.special.factorial(counts_[3], exact=x)
    d_cc = scipy.special.factorial(counts_[4], exact=x)
    d_cd = scipy.special.factorial(counts_[5], exact=x)
    d_dc = scipy.special.factorial(counts_[6], exact=x)
    d_dd = scipy.special.factorial(counts_[7], exact=x)

    # Calculating the cardinality of Alice's contextual set
    cardinality_bob = (n_a*n_b*n_c*n_d/(d_aa*d_ab*d_ba*d_bb*d_cc*d_cd*d_dc*d_dd))
    return cardinality_bob


# This is Upsilon, where summation ranges for l_a1, l_b2, and nu_0_a1_b2 are determined through a guess and check method.
# Each possible combination of these quantum numbers is considered, without taking into account their interdependence.
# This speeds up development time when considering different choices of quantum numbers, but drastically increases computational time.
# The vast majority of trials end in error.
def ontic_state_count(n_, j_, m_a1_, m_b2_, l_a1_, bmap_, s_l_a1_, coefficient_array_):

    # Initialize the ontic state count
    ontic_state_count_ = 0

    # Check to see if we should sum over all values of l_a1
    if s_l_a1_:
        # Define l_a1 range
        l_a1_range_ = np.arange(-(n_ / 2 - j_), (n_ / 2 - j_) + 1, 1)
    else:
        # Define l_a1 range
        l_a1_range_ = np.array([l_a1_])

    # Define the range for l_b2
    l_b2_range_ = np.arange(-(n_ / 2 - j_), (n_ / 2 - j_) + 1, 1)

    # Sum over l_a1 and l_b2
    for i_a in range(len(l_a1_range_)):
        for i_b in range(len(l_b2_range_)):

            # Define DD range
            DD_range_ = np.arange(0, min(j_ - m_a1_, j_ - m_b2_) + 1, 1)

            # Define array to store quantum numbers (to be used with the coefficient array to calculate counts)
            q_number_array = np.ndarray(shape=(8, 8))

            # Define the count array to store the counts associated with each set of quantum numbers
            count_array = []

            # Sum over the path degree of freedom
            for k in range(len(DD_range_)):

                # Fill the quantum number array
                q_number_array[:] = n_, j_, m_a1_, m_b2_, l_a1_range_[i_a], l_b2_range_[i_b], bmap_, DD_range_[k]

                # Calculate the associated counts using the inverted matrix calculated earlier
                count_array.append(np.sum(q_number_array * coefficient_array_, 1))

            # Make sure the count array is valid. This is where we deal with invalid combinations of quantum numbers.
            count_array = np.asarray(count_array)
            count_array_groomed = []

            # Identify situations in which some error has prevented a single set of counts from being defined.
            if len(count_array) > 0:

                # Identify the invalid entries in the count array.
                # Inner condition 1 requires that each count is greater than 0
                # Inner condition 2 requires that each count is an integer
                # If both conditions are met for all 8 counts, the outer condition is met.
                mask = np.where(np.sum(np.where((count_array >= 0) & (count_array % 1 == 0), 1, 0), 1) == 8, 1, 0)

                # Fill a new array with only the sets of counts that were determined to be valid.
                for t in range(len(mask)):
                    if mask[t] == 1:
                        count_array_groomed.append(count_array[t])

                # Identify situations in which there are no valid sets of counts
                if len(count_array_groomed) > 0:
                    count_array_groomed = np.asarray(count_array_groomed, dtype=int)
                    
                    # Define range to iterate over
                    sum_range = len(count_array_groomed)
                    
                    # Calculate the path quantum number nu_0_a1_b2 and the contextual set cardinalities for all valid sets of counts
                    # Initialize Upsilon_a and Upsilon_b
                    Upsilon_a = 0
                    Upsilon_b = 0
                    nu_0_a1_b2 = []
                    for i in range(sum_range):
                        # Calculate the path quantum number
                        nu_0_a1_b2.append((count_array_groomed[i, 0] + count_array_groomed[i, 3] + count_array_groomed[i, 5] + count_array_groomed[i, 6]) / 4)
                        # Calculate the interference term, where the first path quantum number in the array is always the reference value.
                        sign = (-1)**(nu_0_a1_b2[0]-nu_0_a1_b2[i])
                        # Calculate Alice's and Bob's contextual set cardinality and include it in the alternating sums for Upsilon_a and Upsilon_b
                        Upsilon_a += card_alice(count_array_groomed[i])*sign
                        Upsilon_b += card_bob(count_array_groomed[i])*sign

                    # Calculate the product of Upsilon_a and Upsilon_b (i.e. the "Born rule")
                    ontic_state_count_ += Upsilon_a*Upsilon_b

    return ontic_state_count_


# Calculates the probability distribution as a function of theta_ab in the new model
def prob_ssg(n_, j_, m_a1_, m_b2_, l_a1_, s_l_a1_, coefficient_array_):
    # Define arrays that will store data
    probability_data = []
    theta_data = []

    wigner_y_data = []
    wigner_x_data = []

    diff_data = []

    # Initialize b_map at 0
    # b_map is proportional to theta_ab
    bmap_ = 0

    # Sum over all values of b_map, from 0 to n
    while bmap_ <= n_:
        # Print the current value of b_map/n so the user can track progress
        print("B_map= ", bmap_, "/", n)

        # Define the allowed range of m_b2, given n, j, m_a1, and theta_ab (b_map)
        # This is more complicated than simply -j<=m_b2<=j because of the discrete nature of rotation in this model.
        # Note that we could just sum over the full range and invalid quantities would return 0.
        m_b2_min = -j_ + max(0, int(j_ + m_a1_ - bmap_)) + max(0, int(bmap_ - int(2 * (n_/2 - j_)) - j_ - m_a1_))
        m_b2_max = j_ - max(0, int(j_ - m_a1_ - bmap_)) - max(0, int(bmap_ - int(2 * (n_/2 - j_)) - j_ + m_a1_))
        m_b2_range_ = np.arange(m_b2_min, m_b2_max + 1, 1)

        # Initialize the numerator and denominator of our probability calculation
        num = 0
        den = 0

        # Iterate over all allowed values of m_b2
        for q in range(len(m_b2_range_)):
            # Count ontic states
            temp_count = ontic_state_count(n_, j_, m_a1_, m_b2_range_[q], l_a1_, bmap_, s_l_a1_, coefficient_array_)

            # Check to see if the current value of m_b2 is the one the user is interested in.
            # If True, then this count contributes to num and den, if not, it only contributes to den.
            if m_b2_range_[q] == m_b2_:
                num += temp_count
                den += num
            else:
                den += temp_count

        # Verify that something has not gone wrong, leading to den=0.
        # This can happen if the conditioning variables are incompatible with each other.
        if not den == 0:
            # Calculate the probability in the new model
            # print(num)
            # print(den)
            temp_p = num/den

            # Calculate theta
            theta_ = bmap_*math.pi/n_

            # Calculate the probability in QM
            temp_w = wigner_formula(j_, m_a1_, m_b2_, theta_)

            # Record the data for plots
            probability_data.append(temp_p)
            theta_data.append(theta_)
            wigner_y_data.append(temp_w)
            wigner_x_data.append(theta_)

            # Calculate and record the difference between QM and the new model
            diff_data.append(abs(float(temp_p)-temp_w))

        # Advance to the next theta
        bmap_ += 1
    return wigner_x_data, wigner_y_data, theta_data, probability_data, diff_data


# Calculates the probability distribution as a function of theta_ab in QM
def wigner_formula(j_, m_, mp_, theta_):
    # This expression is given in Sakurai (Wigner formula)
    # The conditions are in place to ensure none of the combinatorial arguments become negative.
    q_min = max(0, m_ - mp_)
    q_max = min(j_ - mp_, j_ + m_)
    q = q_min
    wigner = 0

    while q <= q_max:
        if j_ + m_ - q >= 0:
            if q >= 0:
                if mp_ - m_ + q >= 0:
                    if j_ - mp_ - q >= 0:
                        term_1 = mf(j_ + m_) * mf(j_ - m_) * mf(j_ + mp_) * mf(j_ - mp_)
                        term_2 = mf(j_ + m_ - q) * mf(q) * mf(mp_ - m_ + q) * mf(j_ - mp_ - q)
                        term_3 = (math.cos(theta_ / 2) ** (2 * j_ + m_ - mp_ - 2 * q)) * (
                                    math.sin(theta_ / 2) ** (mp_ - m_ + 2 * q))
                        wigner += ((-1) ** (mp_ - m_ + q)) * (math.sqrt(term_1) / term_2) * term_3
        q += 1
    # We then square the result to obtain a probability (Born rule)
    return wigner ** 2


# Execute the calculation and generate plot
if not error:
    data = prob_ssg(n, j, m_a1, m_b2, l_a1, s_l_a1, coefficient_array)


    # Generate plots
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex='all')
    fig.subplots_adjust(hspace=.00)
    axs[0].plot(data[0], data[1], c="blue", label="QM")
    axs[0].plot(data[2], data[3], c="red", label='New Model')
    plot_title_N_l_a1 = "n="+str(n)+r", j="+str(j)+r", $m_{a1}$="+str(m_a1)+r", $m_{b2}$="+str(m_b2)
    plot_title_Y_l_a1 = "n="+str(n)+r", j="+str(j)+r", $m_{a1}$="+str(m_a1)+r", $m_{b2}$="+str(m_b2)+r", $l_{a1}$=$\pm$"+str(abs(l_a1))

    if s_l_a1:
        plot_title = plot_title_N_l_a1

    else:
        plot_title = plot_title_Y_l_a1

    axs[0].set(ylabel='Probability', title=plot_title)
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].legend()

    axs[1].plot(data[0], data[4])
    axs[1].set(xlabel=r'$\theta_{ab}$ (radians)', ylabel=r'|$\Delta$|')
    axs[1].set_xlim(0, math.pi)
    axs[1].set_ylim(-0.005, max(data[4])+.005)

    plt.show()
