Things (intended to be) done by the code in the folder:

1. TTA with BN matching (https://arxiv.org/pdf/2101.10842.pdf) on prostate (SD NCI, TD PROMISE12)
--> Main results summarized in this Google Sheet (https://docs.google.com/spreadsheets/d/1fBJaRqHVScIZZivfjX4Eu3umOJmigvGgH7x-sY2AovI/edit#gid=0)

2. In TTA with BN matching, we adapt the normalization module for each test volume, such that
the means and variances of TI's features become similar to the SD means and variances stored in the BN layers of the network trained on SD.

3. In the evaluate_adaBN.py file, a simpler but similar approach is implemented:
Simply replace the means and variances in the BN layers by the means and variances of the test volume's features.

4. Next, we repeat the two experiments mentioned above on brain images (SD HCPT1, TD ABIDE CALTECH)

5. Next, we will consider the adaBN experiment on simulated contrast / domain shifts
in order to test the hypothesis that ideas like adaBN and TTABN can work only when the domain shifts are affine (x_td = a*x_sd + b).
An idea that is also mentioned in https://arxiv.org/pdf/2103.05898.pdf


