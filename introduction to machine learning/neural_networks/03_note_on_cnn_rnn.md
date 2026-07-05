# Why there's no CNN or RNN demo here

This project deliberately does not include a Convolutional Neural Network
(CNN) or Recurrent Neural Network (RNN) demo, because `data/loan_data.csv`
cannot honestly demonstrate what either one is *for*. Faking a demo on
data these architectures aren't designed for would teach the wrong
intuition, so this note explains what data they actually need instead.

## CNNs need grid-structured data (usually images)

A Convolutional Neural Network's core building block, the convolution
filter, works by sliding a small weight matrix across a *spatial grid*
and looking for local patterns (edges, textures, shapes) that appear
similarly regardless of where in the grid they occur. That only makes
sense when:

- neighboring positions are actually related (the pixel to the right of
  another pixel is spatially adjacent to it), and
- the same pattern can meaningfully appear at different positions
  (a cat's ear looks like a cat's ear whether it's in the top-left or
  bottom-right of the photo).

`loan_data.csv` has 5 unordered columns (age, income, credit_score,
years_employed, existing_debt). There is no "neighboring column" --
`credit_score` isn't spatially adjacent to `years_employed` in any sense
a convolution could exploit, and swapping two columns' order wouldn't
change what the data *means*. A CNN demo here would need genuine
grid/image data, e.g. small greyscale digit images (like MNIST) or any
2D array where nearby cells are actually related.

## RNNs need sequential/temporal data

A Recurrent Neural Network processes a sequence one step at a time,
carrying a hidden state forward so that step *t*'s output can depend on
everything seen at steps 1..t-1. That's built for data where order
matters and there's a variable-length sequence to walk through: words in
a sentence, days in a time series, characters in text (like the
character-level GPT built in the companion `introduction to LLMs`
project).

`loan_data.csv` has exactly one row per applicant with no time
dimension at all -- there's no "next step" for a recurrent hidden state
to carry information toward. Faking an RNN demo would require inventing
a sequence that doesn't exist in this data (e.g. treating the 5 columns
as a fake "sequence" would imply an ordering relationship between age
and credit_score that isn't real). A genuine RNN demo would need actual
sequential data, e.g. an applicant's credit_score sampled monthly over
the past 2 years, or a sequence of past loan applications per person.

## Summary

| Architecture | What it needs | What this dataset has |
|---|---|---|
| CNN | Spatially/grid-related features (images, 2D fields) | 5 unordered scalar columns |
| RNN | An ordered sequence with a time/step dimension | 1 static snapshot per applicant, no time axis |

Every other algorithm in this project works on tabular, single-snapshot
data like this dataset -- that's exactly why CNNs and RNNs are the two
major architectures left out.
