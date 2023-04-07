import scipy.special
import math
from math import factorial as mf
import numpy as np
import matplotlib.pyplot as plt


# Computational time increases with increasing n.
n = 50

# Computational time increases with increasing j and decreasing magnitude of m.
j = 2
m_a1 = 1
m_b2 = 0

# If false, l_a1 will be taken as an additional conditioning variable
# The false condition significantly decreases computational time
s_l_a1 = False

# If s_l_a1 = False, then a value of l_a1 must be chosen. This value must not exceed |(n/2)-j| in magnitude
# This value must always be separated from |(n/2)-j| by an integer. Errors will occur otherwise.
l_a1 = 0

# Coefficient array calculated by Mathematica
# Each row of the coefficient matrix is associated with a count (AA,AB,BA,BB,CC,CD,DC,DD)
# Each column of the matrix is associated with a quantum number (n,j,ma1,mb2,la1,lb2,bmap,DD)
# We use DD rather than mu in this program because its range is simpler to define (see path_counter function)
# The quantum number mu is then defined within the cardinality functions for Alice's and Bob's sub-ensembles
# With all other quantum numbers fixed, each unique choice of DD is equivalent to a unique choice of mu.
coefficient_array = np.asarray([[1/2, 0, -(1/2), -(1/2), 1/2, 1/2, -(1/2), -1],
                                [0, -1, 1/2, 1/2, 1/2, -(1/2), 1/2, 1],
                                [0, -1, 1/2, 1/2, -(1/2), 1/2, 1/2, 1],
                                [1/2, 0, -(1/2), -(1/2), -(1/2), -(1/2), -(1/2), -1],
                                [0, 0, 1, 1, 0, 0, 0, 1],
                                [0, 1, 0, -1, 0, 0, 0, -1],
                                [0, 1, -1, 0, 0, 0, 0, -1],
                                [0, 0, 0, 0, 0, 0, 0, 1]])


# Calculates the cardinality of the ontic states in Alice's and Bob's elementary state spaces (\varepsilon).
# Also calculates mu
def card_alice_vec(bell_c_):
    # Count index: AA=0, AB=1, BA=2, BB=3, CC=4, CD=5, DC=6, DD=7
    # The mu associated with \varepsilon^a
    mu_ = (bell_c_[:, 0] + bell_c_[:, 3] + bell_c_[:, 5] + bell_c_[:, 6])/2

    # x = True bypasses numerical approximations used in scipy.special.factorial. This increases computational time.
    x = False

    # The base-4 counts associated with Alice's event
    A_a1 = bell_c_[:, 0] + bell_c_[:, 1]
    B_a1 = bell_c_[:, 2] + bell_c_[:, 3]
    C_a1 = bell_c_[:, 4] + bell_c_[:, 5]
    D_a1 = bell_c_[:, 6] + bell_c_[:, 7]

    # Cardinality of \varepsilon^a. These are broken into separate terms to avoid overflow as much as possible.
    x_a = scipy.special.factorial(A_a1, exact=x) / (scipy.special.factorial(bell_c_[:, 0], exact=x) * scipy.special.factorial(bell_c_[:, 4], exact=x))
    x_b = scipy.special.factorial(B_a1, exact=x) / (scipy.special.factorial(bell_c_[:, 1], exact=x) * scipy.special.factorial(bell_c_[:, 5], exact=x))
    x_c = scipy.special.factorial(C_a1, exact=x) / (scipy.special.factorial(bell_c_[:, 2], exact=x) * scipy.special.factorial(bell_c_[:, 6], exact=x))
    x_d = scipy.special.factorial(D_a1, exact=x) / (scipy.special.factorial(bell_c_[:, 3], exact=x) * scipy.special.factorial(bell_c_[:, 7], exact=x))

    cardinality_alice = x_a * x_b * x_c * x_d

    return cardinality_alice, mu_


def card_bob_vec(bell_c_):
    # Count index: AA=0, AB=1, BA=2, BB=3, CC=4, CD=5, DC=6, DD=7
    # The mu associated with \varepsilon^b
    mu_ = (bell_c_[:, 0] + bell_c_[:, 3] + bell_c_[:, 5] + bell_c_[:, 6])/2

    # x = True bypasses numerical approximations used in scipy.special.factorial. This increases computational time.
    x = False

    # The base-4 counts associated with Bob's event
    A_b2 = bell_c_[:, 0] + bell_c_[:, 2]
    B_b2 = bell_c_[:, 1] + bell_c_[:, 3]
    C_b2 = bell_c_[:, 4] + bell_c_[:, 6]
    D_b2 = bell_c_[:, 5] + bell_c_[:, 7]

    # Cardinality of  \varepsilon^b. These are broken into separate terms to avoid overflow as much as possible.
    x_a = scipy.special.factorial(A_b2, exact=x) / (scipy.special.factorial(bell_c_[:, 0], exact=x)*scipy.special.factorial(bell_c_[:, 4], exact=x))
    x_b = scipy.special.factorial(B_b2, exact=x) / (scipy.special.factorial(bell_c_[:, 1], exact=x)*scipy.special.factorial(bell_c_[:, 5], exact=x))
    x_c = scipy.special.factorial(C_b2, exact=x) / (scipy.special.factorial(bell_c_[:, 2], exact=x)*scipy.special.factorial(bell_c_[:, 6], exact=x))
    x_d = scipy.special.factorial(D_b2, exact=x) / (scipy.special.factorial(bell_c_[:, 3], exact=x)*scipy.special.factorial(bell_c_[:, 7], exact=x))

    cardinality_bob = x_a * x_b * x_c * x_d

    return cardinality_bob, mu_


# This is Upsilon, where summation ranges for l_a1, l_b2, and mu are determined through a guess and check method.
# Each possible combination of these quantum numbers is considered, without taking into account their interdependence.
# This speeds up development time, but drastically increases computational time.
# The vast majority of trials end in error.
def Upsilon(n_, j_, m_a1_, m_b2_, l_a1_, bmap_, s_l_a1_, coefficient_array_):
    # Many of the unique combinations of quantum numbers being summed over in this function will not be valid.
    # This can be avoided by carefully considering the allowed ranges of the quantum numbers being summed over.
    # Defining these ranges would significantly improve the speed of this calculation, but slows down development.
    # In this code, we initially include the invalid combinations and remove them towards the end.

    # Initialize the count
    product_space_count_ = 0

    # Check to see if we should sum over all values of l_a1, or just use one.
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

            # Define the count array, which stores the counts associated with every complete set of quantum numbers
            count_array = []

            # Sum over the "non-local" degree of freedom (Alice and Bob may disagree).
            for k in range(len(DD_range_)):

                # Fill the quantum number array with the chosen values for the complete set of quantum numbers
                q_number_array[:] = n_, j_, m_a1_, m_b2_, l_a1_range_[i_a], l_b2_range_[i_b], bmap_, DD_range_[k]

                # Calculate the associated counts
                count_array.append(np.sum(q_number_array * coefficient_array_, 1))

            # Make sure the count array is valid. This is where we remove invalid combinations of quantum numbers.
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
                    count_array_groomed = np.asarray(count_array_groomed)

                    # Use the valid sets of counts to calculate the cardinality of Alice's and Bob's sub-ensembles
                    phi_alice_ = card_alice_vec(count_array_groomed)
                    phi_bob_ = card_bob_vec(count_array_groomed)
                    temp_range = len(count_array_groomed)

                    # Sum over all possible combinations of mu_Alice and mu_Bob, accounting for interference.
                    for v in range(temp_range):
                        for u in range(temp_range):
                            product_space_count_ += phi_alice_[0][v]*phi_bob_[0][u]*((-1)**abs((phi_bob_[1][u]-phi_alice_[1][v])/2))

    # return the total number of ordered pairs of ontic states in Alice's and Bob's product space (after interference).
    return product_space_count_


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
        total = 0

        # Iterate over all allowed values of m_b2
        for q in range(len(m_b2_range_)):
            # Get Upsilon
            temp_count = Upsilon(n_, j_, m_a1_, m_b2_range_[q], l_a1_, bmap_, s_l_a1_, coefficient_array_)

            # Check to see if the current value of m_b2 is the one the user is interested in.
            # If True, then this count contributes to num and den, if not, it only contributes to den.
            if m_b2_range_[q] == m_b2_:
                num += temp_count
                den += num
            else:
                den += temp_count

        # Verify that something has not gone wrong, leading to den=0.
        # This can happen if the conditioning variables are incompatible with each other.
        if den != 0:
            # Calculate the probability in the new model
            temp_p = num / den

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
            diff_data.append(abs(temp_p-temp_w))

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
    return (wigner) ** 2


# Execute the calculation and store the data for plotting
data = prob_ssg(n, j, m_a1, m_b2, l_a1, s_l_a1, coefficient_array)


# Generate plots
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex='all')
fig.subplots_adjust(hspace=.00)
axs[0].plot(data[0], data[1], c="blue", label="QM")
axs[0].plot(data[2], data[3], c="red", label='New Model')
plot_title_N_l_a1 = "n="+str(n)+r", j="+str(j)+r", $m_{a1}$="+str(m_a1)+r", $m_{b2}$="+str(m_b2)
plot_title_Y_l_a1 = "n="+str(n)+r", j="+str(j)+r", $m_{a1}$="+str(m_a1)+r", $m_{b2}$="+str(m_b2)+r", $l_{a1}$=+$\pm$"+str(abs(l_a1))

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
