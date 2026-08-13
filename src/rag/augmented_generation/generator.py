from pathlib import Path
from ..retriever.retriever import retriever
from google import genai
from pydantic import BaseModel


class Claim(BaseModel):
    claim: str
    page_ids: list[int]

    # We define a class that inherits from the BaseModel class. Once it inherits it the class can then create Pydantic models
    # This class is the "single source of truth" for validation — it's used TWICE:
    #   1) converted into a JSON schema and sent to Gemini so it knows the exact
    #      shape/types to generate output in
    #   2) used again on the way back to actually validate + parse Gemini's raw
    #      JSON string into real Python objects

    # Pydantic doesn't just check "is this valid JSON" — it checks TYPES too.
    # e.g. if Gemini returns page_ids as ["1", "2"] (strings) instead of [1, 2]
    # (ints), Pydantic will try to coerce them to int, and if it can't, it'll
    # raise a ValidationError. So this is stricter than plain json.loads().



def build_user_prompt(query, user_template, hf_access_token, cluster_url, cluster_api_key):

    retr_list=retriever(query=query, hf_access_token=hf_access_token, cluster_url=cluster_url, cluster_api_key=cluster_api_key)

    if len(retr_list)!=0:

        context_string=''

        for i in retr_list:

            id=i['id']
            source_temp=i['payload']['source']
            source=source_temp.split('\\')[-1].replace('.pdf','')
            page_content=i['payload']['page_content'].replace(r'\n',' ').strip()

            if retr_list.index(i)==len(retr_list)-1:
                context_temp=fr'Page ID: {id}, Page source: {source}, Page content: {page_content}'
            else:
                context_temp=fr'Page ID: {id}, Page source: {source}, Page content: {page_content}'+'\n\n'

            context_string+=context_temp

        query_final_temp=user_template.replace("[QUERY]",query)
        query_final=query_final_temp.replace("[SOURCES]",context_string)

    else:

        query_final='EMPTY'

    return query_final



def generator(prompt, system_prompt, gemini_api_key, generator_model='gemini-3.6-flash'):

    if prompt == 'EMPTY':
        output = 'No relevant documents found to the provided query'
        # Note: in this branch `output` is just a plain string, NOT a list[Claim]. No validation happens here since we never call Gemini — worth keeping in mind downstream if calling code assumes `output` is always list[Claim].

    else:
        gemini_client = genai.Client(api_key=gemini_api_key)

        interation = gemini_client.models.generate_content(
            model=generator_model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=list[Claim]))

        output: list[Claim] = interation.parsed

    return output


# Step 1 (model-side, format only): tells Gemini "your raw output must be valid JSON text" — doesn't constrain shape at all.

# Step 2 (model-side, shape): Pydantic model gets converted into a JSON Schema under the hood and passed to Gemini alongside the
# prompt. This schema gets sent to the Gemini API alongside the prompt essentially as extra instructions the model is told to follow.
# This is what actually constrains Gemini to produce an ARRAY of objects, each with a "claim" (string) and "page_ids"
# (list of ints) key. Gemini is instructed to conform to this schema while generating — but this is a "best effort" guide for
# the model, not a guarantee; the model could still technically produce something that fails the check in Step 3.


# Concretely: at each step of generating output, an LLM produces a probability distribution over possible next tokens. 
# Normally it just samples from that full distribution. With constrained decoding, the API takes your JSON Schema, 
# compiles it into something like a formal grammar, and at every single token step it masks out (sets to zero probability) 
# any token that would make the output stop conforming to that grammar. So the model is literally not able to emit a 
# token like "page_ids": "hello" (a string where a number-array is required) — that path is blocked before it can be chosen. 
# It can only pick among the tokens that keep it inside a syntactically-and-structurally valid path toward your schema. 
# This is why it's much stronger than just "asking nicely via prompt": if a plain prompt said "please output JSON matching '
# 'this schema," the model could still hallucinate a wrong field name or wrong type. Constrained decoding makes the wrong output 
# essentially unreachable at the character/token level for anything the grammar covers — mainly structure and value type/format 
# (string vs. number vs. array vs. object, required keys present, etc.).


# Step 3 (client-side, actual validation): this is where real validation happens. `.parsed` does NOT just return `.text` cast to a Python list — under the hood the SDK:
#   a) takes the STRING containing the required json output Gemini generated (accessible separately via `interation.text` if you want to see the unvalidated version)
#   b) json.loads() it into raw Python list/dicts
#   c) feeds that into Claim.model_validate() (Pydantic) for EACH item
#   d) if any item fails (wrong type, missing field, etc.), this step raises an error rather than silently returning bad data
#   e) on success, returns a list of actual Claim INSTANCES (not dicts) — so downstream you can do `output[0].claim` / `output[0].page_ids`
#      with attribute access, autocomplete, and type safety, rather than output[0]['claim'] with a plain dict.

# application_mime_type defines the format in which gemini is supposed to return an output.
# NOTE: The output, even though is formatted into the mentioned mime type, is still a string, unless we validate it beforehand (like we have done here)
# response_schema_type defines the schema of every json output, giving it a proper structure so that llm does not vary between output structures, due to its proailistic nature.

# we could have written output=interaction.parsed, but output:list[Claim] says that whatever output we get, it must be of the form list of Claim type object (We defined the Claim data type in the top class)
# This ensures that the STRING containing the required json output generated by the model is ready to be used as a python json object (we can then use that readymade json elsewhere as well, without havig through go through the 
# effort of converting it into one first)



def retrieval_augmented_generation(query, user_template, system_prompt, hf_access_token, cluster_url, cluster_api_key, gemini_api_key, generator_model= 'gemini-3.6-flash'):

    user_prompt=build_user_prompt(query, user_template, hf_access_token, cluster_url, cluster_api_key)
    output=generator(user_prompt, system_prompt, gemini_api_key, generator_model=generator_model)

    return output




if __name__=='__main__':

    PARENT=Path(__file__).resolve().parent
    SRC=PARENT.parent
    ROOT=SRC.parent

    USER_TEMPLATE=ROOT/'src'/'rag'/'augmented_generation'/'prompt'/'user_template.txt'
    SYSTEM_TEMPLATE=ROOT/'src'/'rag'/'augmented_generation'/'prompt'/'system_template.txt'
    SAMPLE_OUTPUT=ROOT/'src'/'rag'/'augmented_generation'/'samples'/'prompts'/'sample_augmented_generation_output.txt'

    import os
    from dotenv import load_dotenv

    load_dotenv()

    hf_access_token=os.getenv('HF_ACCESS_TOKEN')
    qdrant_cluster_url=os.getenv('QDRANT_URL')
    qdrant_api_key=os.getenv('QDRANT_API_KEY')
    gemini_api_key=os.getenv('GEMINI_API_KEY')

    with open(USER_TEMPLATE, 'r') as f:
        user_template=f.read()

    with open(SYSTEM_TEMPLATE, 'r') as f:
        system_prompt=f.read()

    query=input('What is your question: ')

    out=retrieval_augmented_generation(query=query, 
                             user_template=user_template, 
                             system_prompt=system_prompt, 
                             hf_access_token=hf_access_token, 
                             cluster_url=qdrant_cluster_url, 
                             cluster_api_key=qdrant_api_key,
                             gemini_api_key=gemini_api_key)

    with open(SAMPLE_OUTPUT,'w', encoding='utf-8') as f:
        f.write(out)

    print(out)





    

