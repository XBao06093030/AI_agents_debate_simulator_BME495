# Simulate a Debate
#
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import LLMChain

# 0. Key skill
  # You need to know who your opponents are and nicely designed the prompt.


# 1. Define the topic

topic = "Biomedical engineering has experienced significant development in recent years.\
People believe that human beings are going to master gene-editing technologies in the near future.\
Some people feel excited about this new technology, which may radically cure many diseases and overcome some naturally born flaws.\
Some people dislike this technology because it may cause problems in the scope of editing safety, unknown risks, social fairness, etc."


# 2. Create a custom prompt template

agent1_system = SystemMessagePromptTemplate.from_template(
    topic + "You are Agent 1. You strongly believe that gene editing can bring more benefits than potential loss."
    "Rebut the points your opponent agent (Agent 2) makes."
    #Fine-designed prompt
    "If Agent2 argues that gene editing is not safe, you can rebuttal that nothing is 100% safe. A plane may crash. \
    Why do people still ride it? With the development of technology, \
    it will be safer and safer and keep the risk at our tolerance level."

    "Agent 2 may argue that gene editing is not fair. Rich people may have access to higher technology, \
    which can make them better than poor people and cause more social unfairness. Then you can rebuttal that nothing is 100% fair in\
    human society. Rich people can feed their kids steak in every meal, and poor people may only have bread. \
    If the kids compete in a sport, rich kids are more likely to win. \
    Therefore, we can see that the social unfairness is caused by the resource allocation mechanism, not gene editing."

    #Add more if you need
)

agent2_system = SystemMessagePromptTemplate.from_template(
    topic + "You are Agent 2. You strongly believe that gene editing is dangerous and evil. "
    "Rebut the points your opponent agent (Agent 1) makes."
    #Fine-designed prompt
    "It may cause unknown risks, which may make mankind extinct. \
    It may also cause unfairness because rich people can edit them better, \
    further widening the economic status gap between the rich and poor. "

    "If Agent 1 argues that the nothing is 100% fair in\
    human society. Rich people can feed their kids steak in every meal, and poor people may only have bread. \
    If the kids compete in a sport, rich kids are more likely to win. \
    Therefore, we can see that the social unfairness is caused by the resource allocation mechanism, not gene editing.\
    You may consider arguing that the gap caused by the editing can strengthen the ability and motivation of the rich to keep the gap. \
    It may make it even harder for the poor to catch up."
    
    #Add more if you need
)


agent1_user = HumanMessagePromptTemplate.from_template(
    "Agent 2 says: {opponent_argument}\n\n"
    "Agent 1, please respond with your rebuttal and any new points supporting gene editing."
)

agent2_user = HumanMessagePromptTemplate.from_template(
    "Agent 1 says: {opponent_argument}\n\n"
    "Agent 2, please respond with your rebuttal and any new points opposing to gene editing."
)


agent1_prompt = ChatPromptTemplate.from_messages([agent1_system, agent1_user])
agent2_prompt = ChatPromptTemplate.from_messages([agent2_system, agent2_user])


# 3. Create an LLMChain with ChatOpenAI and your custom prompt

agent1_chain = LLMChain(
    llm=ChatOpenAI(
        temperature=0.1,  # increase if you want more variation
        model_name="gpt-3.5-turbo", # this is a free model and we don't have financial support for chargeable model now        
        openai_api_key="YOUR_OPENAI_API_KEY"
    ),
    prompt=agent1_prompt
)

agent2_chain = LLMChain(
    llm=ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",         
        openai_api_key="YOUR_OPENAI_API_KEY"
    ),
    prompt=agent2_prompt
)



# 4. Simulation of debate starts
NUM_ROUNDS = 3

# We'll keep track of each agent's last argument so we can feed it to the other.
agent1_argument = "I support gene editting. This is the future of human being society."
agent2_argument = "I have objection to gene editting. It is dangerous and evil"

print("=== STARTING DEBATE ===")
print(f"\n[Agent 1, Initial Statement]\n{agent1_argument}")
print(f"\n[Agent 2, Initial Statement]\n{agent2_argument}")

for round_index in range(1, NUM_ROUNDS + 1):
    # Agent 1 responds to Agent 2's last argument:
    agent1_response = agent1_chain.run(opponent_argument=agent2_argument)
    print(f"\n[Round {round_index} - Agent 1]:\n{agent1_response}")
    agent1_argument = agent1_response  # update Agent1's last statement

    # Agent 2 responds to Agent 1's last argument:
    agent2_response = agent2_chain.run(opponent_argument=agent1_argument)
    print(f"\n[Round {round_index} - Agent 2]:\n{agent2_response}")
    agent2_argument = agent2_response  # update Agent2's last statement

