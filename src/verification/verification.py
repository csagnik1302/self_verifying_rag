import pysbd
import re


def output_sentence_parser(text):

    seg=pysbd.Segmenter(language='en', clean=False)
    sentences=seg.segment(text)

    out=[i for i in sentences]

    return out



def id_extractor(output_sent_list):

    output_dict={}

    for i in output_sent_list:

        ids=re.findall(r'\[(\d+)\]',i)
        current_sentence=re.sub(r'\s*\[(\d+)\]','',i)

        output_dict[current_sentence.strip()]=list(int(j) for j in ids)
    
    return output_dict


test_cases = [
    # Basic case — should split into exactly 2 sentences, clean citations
    "The 2014 FIFA World Cup final was contested between Germany and Argentina, [475]. "
    "Argentina lost the final and finished as runner-up [475][592].",

    # Abbreviation before a citation — "Gen." should NOT trigger a split
    "Gen. Jan Smuts ordered the release of political prisoners in January 1914 [12]. "
    "This followed extensive non-violent protest led by Gandhi [12][13].",

    # Decimal number right next to a citation tag
    "Revenue grew 12.5% year-over-year to $450.3M in Q3 [7]. "
    "Net profit margin improved to 8.1% from 6.4% [7].",

    # Multiple abbreviations in one sentence
    "Dr. B.R. Ambedkar chaired the drafting committee of the Indian Constitution [88]. "
    "The U.S. and U.K. governments later cited it as a model [88][91].",

    # Sentence with NO citation at all — should still be extracted, flagged unsupported
    "Some historians dispute the exact timeline of these events. "
    "Gandhi returned to India on 9 January 1915 [5].",

    # Citation tag immediately after a decimal-formatted date
    "The treaty was signed on 15.8.1947 [201]. "
    "It marked the formal transfer of power [201][202].",

    # Three citations stacked on one sentence
    "The independence movement drew support from multiple regions and communities [1][2][3]. "
    "Its influence extended well beyond South Asia [3].",

    # Worst-case adjacency: double abbreviation right before a citation
    "The report was submitted to the Rt. Hon. Governor-General [44]. "
    "No further action was taken that year [44].",

    # Ellipsis / quote inside a sentence
    "The declaration stated the nation was \"free and sovereign...\" as of that date [99]. "
    "Celebrations followed across major cities [99][100].",

    # Single-sentence input, no trailing period before EOF
    "Independence was formally declared at midnight on 15 August 1947 [150]"
]

for i in test_cases:

    parsed_output=output_sentence_parser(i)
    dict=id_extractor(parsed_output)

    print(dict)