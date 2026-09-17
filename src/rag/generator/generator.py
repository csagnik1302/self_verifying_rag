import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import tomllib
from pathlib import Path


#
import sys
from pathlib import Path

ORC=Path(__file__).resolve().parent
RAG=ORC.parent
SRC=RAG.parent
ROOT=SRC.parent

utils_dir=ROOT/'src'/'rag'/'utils'

sys.path.insert(0,str(utils_dir))
from utils import State
#


load_dotenv()

with open('config.toml','rb') as f:
    config=tomllib.load(f)


generator_model_name=config['generator']['model']

client=genai.Client(api_key=os.getenv('GEMINI_API_KEY'))


GENERATOR=Path(__file__).resolve().parent
RAG=GENERATOR.parent
SRC=RAG.parent
ROOT=SRC.parent

with open(file=ROOT/'src'/'rag'/'generator'/'prompt_template'/'system_prompt.txt', mode='r', encoding='utf-8') as f:
    SYSTEM_PROMPT=f.read()

def prompt_builder(query, rel_docs):

    starting='QUESTION: '+query+'\n\n'+'CONTEXT:'+'\n'

    for i in range(len(rel_docs)):
        starting+=f'[P{i+1}] {rel_docs[i]['text']}'+'\n'

    return starting



def response_generation(prompt, system_prompt=SYSTEM_PROMPT, client=client, model=generator_model_name):

    response=client.models.generate_content(model=model,
                                               config=types.GenerateContentConfig(system_instruction=system_prompt),
                                               contents=prompt)

    return response.text



def generator(state:State) -> State:
    prompt=prompt_builder(query=state['query'], rel_docs=state['rerank_docs'])
    response=response_generation(prompt=prompt)
    return {'response':response}


if __name__=='__main__':

    query='What laws are there against deepfakes'
    retr_set=[{'text': 'and subjected to stricter regulation. However, the Act does not explicitly mandate specific\ndeepfake detection mechanisms, nor does it comprehensively address the broader human\nrights implications stemming from the proliferation of such synthetic media. Scholars have\nfurther argued that manufacturer self-assessments are inadequate to address complex\ngenerative harms, necessitating mandatory architectural mandates and standardized\nverification frameworks to ensure effective enforcement.\nPenalties\nNon-compliance with theprohibitions in Article 5 is subject to administrative fines of up to\nEUR 35,000,000 or, if the offender is an undertaking, up to 7% of its total worldwide annual\nturnover, whichever is higher. Other operator obligations may be sanctioned with fines of up\nto EUR 15,000,000 or 3% of worldwide annual turnover, whichever is higher; providing\nincorrect, incomplete or misleading information may be fined up to EUR 7,500,000 or 1% of', 'rerank_score': 5.662228107452393}, {'text': "Deepfake detection and regulation\nLegal experts are actively questioning whether current and emerging regulatory frameworks\nadequately balance the advancements in deepfake detection with the protection of individual\nrights. Relevant legislation being scrutinized includes the EU AI Act, the General Data\nProtection Regulation (GDPR), the Digital Services Act in the European Union, as well as the\nfragmented state and federal laws in the United States, the Online Safety Act 2023 in the\nUnited Kingdom, and China's Administrative Provisions on Deep Synthesis in Internet-Based\nInformation Services (commonly known as the Deep Synthesis Provisions). Scholars are\nevaluating if these frameworks effectively address the complex interplay between technology,\nrights, and responsibilities in the context of deepfakes.\nPrevention\nHenry Ajder who works for Deeptrace, a company that detects deepfakes, says there are\nseveral ways to protect against deepfakes in the workplace. Semantic passwords or secret", 'rerank_score': 2.832115650177002}, {'text': "representative for New York's 9th congressional district Yvette Clarke.\nIn 2024, over halfof documented identity fraud involved AI-created forgeries, leading several\nstates to introduce legislation regarding deepfakes, including Virginia, Texas,California, and\nNew York; charges as varied as identity theft, cyberstalking, and revenge porn have been\npursued, while more comprehensive statutes are urged.\nAmong U.S. legislative efforts, on 3 October 2019, California governor Gavin Newsom signed\ninto law Assembly Bills No. 602 and No. 730. Assembly Bill No. 602 provides individuals\ntargeted by sexually explicit deepfake content made without their consent with acause of\naction against the content's creator. Assembly Bill No. 730 prohibits the distribution of\nmalicious deepfake audio or visual media targeting a candidate running for public office\nwithin 60 days of their election. U.S. representative Yvette Clarke introduced H.R. 5586:", 'rerank_score': 1.8297197818756104}, {'text': 'TRAIGA, or the Texas Responsible Artificial Intelligence Governance Act, is a state law\nregulating the development and deployment of artificial intelligence (AI) systems in Texas.\nSponsored by Representative Giovanni Capriglione, the Act establishes a framework\ngoverning certain uses of AI, outlines prohibited uses, and creates obligations on state\ngovernment entities, among other provisions. TRAIGA was signed into law in 2025 and took\neffect on January 1, 2026. \nThe law applies to AI developers and deployers that conduct business in Texas or whose\nsystems are used by Texas residents. It prohibits the intentional development or deployment\nof AI systems to incite harm, violate constitutional rights, engage in unlawful discrimination,\nand produce child sexual abuse material or unlawful deepfakes. TRAIGA also establishes the\nTexas Artificial Intelligence Council and creates a regulatory sandbox program. The Texas\nAttorney General is charged with enforcement.', 'rerank_score': 0.47923076152801514}, {'text': "employees, unless the tools have been independently audited for bias.\nThe Responsible AI Safety and Education Act (RAISE Act) is a New York State law that imposes\ntransparency, safety, and reporting requirements on developers of large frontier artificial\nintelligence models. The law was signed by Governor Kathy Hochul on December19, 2025. It\nis expected to take effect on January 1, 2027.\nTennessee\nOn March 21, 2024, the State of Tennessee enacted legislation called the ELVIS Act, aimed\nspecifically at audio deepfakes, and voice cloning. This legislation was the first enacted\nlegislation in the nation aimed at regulating AI simulation of image, voiceand likeness.  The\nbill passed unanimously in the Tennessee House of Representatives and Senate. This\nlegislation's success was hoped by its supporters to inspire similar actions in other states,\ncontributingto a unified approach to copyright and privacy in the digital age, and to reinforce", 'rerank_score': 0.08280393481254578}, {'text': "Johansson.\nSeveral incidents involving sharing of non-consensual deepfake pornography have occurred.\nIn late January 2024, deepfake images of American musician Taylor Swift proliferated. Several\nexperts have warned that deepfake pornography is more quickly created and disseminated,\ndue to the relative ease of using the technology. Canada introduced federal legislation\ntargeting sharing of non-consensual sexually explicit AI-generated photos; most provinces\nalready had such laws. In the United States, the DEFIANCEAct was introduced in March 2024.\nEnvironment\nA large amount of electricity is needed to power generative AI products, making it more\ndifficult for companies to achieve net zero emissions. From 2019 to 2024, Google's\ngreenhouse gas emissions increased by nearly 50%, partly as a result of increased energy\nconsumption by AI data centres.\nBiosecurity and cybersecurity", 'rerank_score': -0.586916983127594}, {'text': 'Chat site Discord took action against deepfakepornography in 2018, and has taken a general\nstance against deepfakes. Gfycat began removing all deepfakes from its site on 31 January\n2018.\nReddit banned the r/deepfakes subreddit on 7 February 2018, due to the policy violation of\n"involuntary pornography". Also in February 2018, Pornhub said that it would ban deepfake\nvideos on its website because it is considered "non consensual content" which violates their\nterms of service. They had also stated previously that they will take down content flagged as\ndeepfakes. Writers from Motherboard reported that searching "deepfakes" on Pornhub still\nreturned multiple recent deepfake videos.\nGoogle added "involuntary synthetic pornographic imagery" to its ban list in September 2018,\nallowing anyone to request the block of results showing their fake nudes. In May 2022, Google', 'rerank_score': -0.5875834226608276}, {'text': 'information via the Internet.\nSee also\nReferences\nFurther reading\nDaniel Immerwahr, "Your Lying Eyes: People now use A.I. togenerate fake videos\nindistinguishable from real ones. How much does it matter?", The New Yorker, 20 November\n2023, pp. 54–59. "If by \'deepfakes\' we mean realistic videos produced using artificial\nintelligence that actually deceive people, then they barely exist. The fakes aren\'t deep, and the\ndeeps aren\'t fake. [...] A.I.-generated videos are not, in general, operating in our media as\ncounterfeited evidence. Their role better resembles that of cartoons, especially smutty ones."\n(p. 59.)\nEmmanouil Billis, "Deepfakes και Ποινικό Δίκαιο [Deepfakes and the Criminal Law]"(in Greek).\nIn: H. Satzger et al. (eds.), The Limits and Future of Criminal Law - Essays in Honor of Christos\nMylonopoulos, Athens, P.N. Sakkoulas, 2024, pp. 689–732.\nExternal links\nSasse, Ben (19 October 2018). "This New Technology Could Send American Politics into a', 'rerank_score': -0.6470045447349548}, {'text': 'the fake videos are non-consensual pornography. Most of the victims of these videos were\ncelebrities or high-profile individuals.\nIn February 2018, r/deepfakes was banned by Reddit for sharing involuntary pornography.\nOther websites have also banned the use of deepfakes for involuntary pornography, including\nthe social media platform Twitter and the pornography site Pornhub. However, some websites\nhave not yet banned Deepfake content, including 4chan and 8chan.\nNon-pornographic deepfake content continues to grow in popularity with videos from\nYouTube creators such as Ctrl Shift Face and Shamook. A mobile application, Impressions, was\nlaunched for iOS in March 2020. The app provides a platform for users to deepfake celebrity\nfaces into videos in a matter of minutes.\nImage synthesis\nImage synthesis is the artificial production of visual media, especially through algorithmic\nmeans. In the emerging world of synthetic media, the work of digital-image creation—once', 'rerank_score': -0.651620626449585}]

    print(generator(query=query, rel_docs=retr_set))