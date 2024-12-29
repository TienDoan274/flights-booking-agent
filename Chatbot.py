from dotenv import load_dotenv
# set your OPENAI_API_KEY in .env file
load_dotenv()

from llama_index.llms.openai import OpenAI
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    step, 
    Event,
    Context,
    Workflow,
    draw_all_possible_flows,
    WorkflowCheckpointer
)
import dateparser
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.prompts import PromptTemplate
from pymongo import MongoClient

class DatabaseSchema(BaseModel):
    flight_code: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    departure_location: Optional[str] = None
    arrival_location: Optional[str] = None
    airline: Optional[str] = None
    ticket_price: Optional[int] = None
    
class MongoSchema(BaseModel):
    date:Optional[str] = None
    scheduled_time:Optional[str] = None
    updated_time:Optional[str] = None
    #route:Optional[str] = None
    flight_id:Optional[str] = None
    counter:Optional[str] = None
    gate:Optional[str] = None
    status:Optional[str] = None
'''
schema = StructType([
        StructField("date", StringType(), True),
        StructField("scheduled_time", StringType(), True),
        StructField("updated_time", StringType(), True),
        StructField("route", StringType(), True),
        StructField("flight_id", StringType(), True),
        StructField("counter", StringType(), True),
        StructField("gate", StringType(), True),
        StructField("status", StringType(), True)
    ])
'''


class ConnectDB_Event(Event):
    payload: str 

class ConnectDBComplete_Event(Event):
    payload: str

class ParseInput_Event(Event):
    payload: str

class QueryGeneration_Event(Event):
    payload: str

class PostProcessQuery_Event(Event):
    payload: str

class QueryGenerationComplete_Event(Event):
    payload: str
    
class RetrieveEvent(Event):
    payload: str

class CleanUp(Event):
    payload: str

import json

# Custom JSON encoder for ObjectId
from bson import ObjectId
class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)


class MongoDBflow(Workflow):
    @step 
    async def start(self, ctx: Context ,ev: StartEvent) -> ConnectDB_Event | QueryGeneration_Event :

        MONGO_DB = "quickstart"  # Database name
        COLLECTION_NAME = "topicData"  # Collection name
        CONNECTION_STRING = "mongodb://localhost:35001" # connection string

        LLM = OpenAI(model='gpt-3.5-turbo', temperature=0)
    
        await ctx.set('MONGO_DB', MONGO_DB)
        await ctx.set('COLLECTION_NAME', COLLECTION_NAME)
        await ctx.set('CONNECTION_STRING', CONNECTION_STRING)
        await ctx.set('LLM', LLM)

        try:
        # Replace single quotes with double quotes to make it valid JSON
            query_string = ev.query.replace("'", '"')
            db_query = json.loads(query_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid query format: {ev.query}") from e

        print(f'Initial query: {db_query}')
        await ctx.set('mongoDB_query', db_query)

        ctx.send_event(ConnectDB_Event(payload=''))
        ctx.send_event(QueryGenerationComplete_Event(payload=''))

    @step
    async def checking_query(self, ctx: Context, ev: QueryGeneration_Event) -> QueryGenerationComplete_Event:
        # PERFORM CHECKING VALID QUERY
        if not isinstance(ev.payload, dict):
            raise ValueError("Input is not valid query object")
        
        #ctx.send_event(QueryGenerationComplete_Event(payload='Query generated success'))
        return QueryGenerationComplete_Event(payload='Query generated success')
    
    @step
    async def connect_mongoClient(self, ctx: Context, ev: ConnectDB_Event) -> ConnectDBComplete_Event:
        try:
            connection_string = await ctx.get('CONNECTION_STRING')
            mongo_db = await ctx.get('MONGO_DB')
            collection_name = await ctx.get('COLLECTION_NAME')

            client = MongoClient(connection_string)
            await ctx.set('mongo_client' , client)

            db = client.get_database(name=mongo_db)
            collection = db[collection_name]

            await ctx.set('collection', collection)
            return ConnectDBComplete_Event(payload='Connect to database success')
        except Exception as e:
            raise ValueError(f"Failed to connect to MongoDB: {str(e)}")

    @step
    async def retrieve_mongoClient(self, ctx: Context, ev: ConnectDBComplete_Event | QueryGenerationComplete_Event ) -> CleanUp:
        
        #print("Received event: ", ev.payload)
        if ( ctx.collect_events( ev, [ConnectDBComplete_Event, QueryGenerationComplete_Event]) is None):
            return None
        try: 
            collection = await ctx.get('collection')
            mongoDB_query = await ctx.get('mongoDB_query')
            final_query = {}
            ### Apply retrieving logic with schemas
            for key, value in mongoDB_query.items():
                if key == 'departure_time' or key == 'arrival_time':
                    final_query[key] = { "$regex": f"^{value}"}
                else:
                    final_query[key] = value
            
            print(f'Final Query: {final_query}')
            retrieved_data = collection.find(final_query).to_list(length=None)
            await ctx.set('retrieved_data', retrieved_data)

            return CleanUp(payload='Retrieving data success')
        except Exception as e:
            raise ValueError(f"Failed to retrieve data: {str(e)}")

    @step
    async def clean_up(self, ctx: Context, ev: CleanUp) -> StopEvent:
        client = await ctx.get('mongo_client', None)
        retrieved_data = await ctx.get('retrieved_data', None)

        if not retrieved_data: # empty list
            retrieved_data = 'No data retrieved, tell user there is no flights available.'
        if client:
            client.close()
            print("MongoDB client closed.")
        
        formatted_string = "\n".join(json.dumps(item, indent=4, cls=MongoEncoder) for item in retrieved_data)
        
        print(formatted_string)
        observation = f"Retrieved data: \n{formatted_string}"
        return StopEvent(result=observation)

#draw_all_possible_flows(MongoDBflow)
# - `route` (string): The flight's route, in the format **"from <destination1> to <destination2>"**. Both `destination1` and `destination2` must be provided. For example, "from NYC to LAX".
# "route": "from NYC to LAX",

class GatherInformation(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> ParseInput_Event:
        LLM = OpenAI(model='gpt-3.5-turbo', temperature=0)
        await ctx.set('LLM', LLM)
        print(f'User input: {ev.query}')
        
        await ctx.set('user_query', ev.query)

        return ParseInput_Event(payload= ev.query)
    
    @step
    async def parse_userQuery(self, ctx: Context, ev: ParseInput_Event) -> QueryGeneration_Event:

    simplified_prompt = PromptTemplate(
    """
    You are a helpful flight attendant. If you encounter time-related terms like 'yesterday', 'tomorrow', 'today', or a specific date, extract them as they are and map them to `departure_time` or `arrival_time` where applicable.
    
    Extract the following information from the query below. If any field is not explicitly mentioned in the query, set it as `null`. Follow the schema strictly for the JSON response:
    
    **Schema:**
    - `date` (string): The date associated with the flight, including terms like 'today', 'tomorrow', or specific dates (e.g., '2024-12-25').
    - `scheduled_time` (string): The scheduled time of the flight, if mentioned (e.g., '14:30').
    - `updated_time` (string): The updated or revised time, if applicable.
    - `flight_id` (string): The flight's identifier.
    - `counter` (string): The check-in counter details, if provided.
    - `gate` (string): The gate number for the flight.
    - `status` (string): The flight status'.

    Example Query:
    "What is the status of flight AB123 from NYC to LAX tomorrow at 14:30?"

    Example Response:
    ```json
    {
        "date": "tomorrow",
        "scheduled_time": "14:30",
        "updated_time": null,
        "flight_id": "AB123",
        "counter": null,
        "gate": null,
        "status": null
    }
    ```

    Now, process the following query and respond as a JSON object:

    Query: {text}

    Respond:
    """
    )
        try:
            llm = await ctx.get('LLM')
            user_query = await ctx.get('user_query')
            extracted_input = llm.structured_predict( MongoSchema, simplified_prompt, text= ev.payload)

            print(f"Extracted input: {extracted_input}")
            
            await ctx.set('extracted_input', extracted_input)
            return QueryGeneration_Event(payload='')
        
        except Exception as e:
            raise ValueError(f"Failed to parse query: {str(e)}")
        
    @step
    async def generate_query(self, ctx: Context, ev: QueryGeneration_Event) -> PostProcessQuery_Event:
        
        #pydantic object
        parsed_input = await ctx.get('extracted_input')

        if not isinstance(parsed_input, BaseModel):
            raise ValueError("Parsed input is not a valid Pydantic object.")
    
        # Assuming 'model' is your Pydantic object
        mongoDB_query = parsed_input.model_dump(exclude_none=True)

        if not isinstance(mongoDB_query, dict):
            raise ValueError("Generated query is not a valid MongoDB query object.")
        
        await ctx.set('mongoDB_query', mongoDB_query)

        print(f"Initial MongoDB query: {mongoDB_query}")
        return PostProcessQuery_Event(payload="Generated Initial Query success")
        
    
    @step
    async def postprocess_query(self, ctx: Context, ev: PostProcessQuery_Event) -> QueryGenerationComplete_Event:
        
        def parseTime(time: str):
            parsed = dateparser.parse(time).strftime("%Y-%m-%d")
            '''
            if parsed.hour == 0 and parsed.minute==0:
                parsed = parsed.strftime("%Y-%m-%d")
            '''

            return parsed

        #print(ev.payload)
        tmp = await ctx.get('mongoDB_query')

        '''
        "departure_time": departure_time.strftime("%Y-%m-%d %H:%M"),
        "arrival_time": arrival_time.strftime("%Y-%m-%d %H:%M")
        '''
        departTime = tmp.get('scheduled_time', None)  # Default to None if the key doesn't exist
        arrivalTime = tmp.get('updated_time', None)

        # do the query processing here
        if departTime is not None:
            parsedDepart = parseTime(departTime)
            tmp['scheduled_time'] = parsedDepart
        if arrivalTime is not None:
            parsedArrival = parseTime(arrivalTime)
            tmp['updated_time'] = parsedArrival

        mongoDB_query = tmp
        
        await ctx.set('mongoDB_query', mongoDB_query)
        print(f"Final MongoDB query: {mongoDB_query}")

        return QueryGenerationComplete_Event(payload='')


    @step
    async def query_output(self, ctx: Context, ev: QueryGenerationComplete_Event) -> StopEvent:
        
        mongoDB_query = await ctx.get('mongoDB_query')

        observation = f'MongDB query: {mongoDB_query}.'
        return StopEvent(result=observation)

async def prepare_input(user_query: str) -> dict:
    w = GatherInformation(timeout=30, verbose=False)
    result = await w.run(query= user_query)
    return result

    
async def read_mongodb(mongodb_query: str) -> str: 
    
    w = MongoDBflow(timeout=30, verbose=False)
    result = await w.run(query= mongodb_query)
    return result




from llama_index.agent.openai import OpenAIAgent
import nest_asyncio
nest_asyncio.apply()

async def main():
    # Define the MongoDB Retriever tool
    read_mongodb_tool = FunctionTool.from_defaults(
        async_fn=read_mongodb,
        name='MongoDB_Retriever',
        description=(
            'Tool to retrieve information from the MongoDB database. This tool is used after generating a valid MongoDB query.'
        )
    )

    # Define the Query Preparation tool
    prepare_input_tool = FunctionTool.from_defaults(
        async_fn=prepare_input,
        name='Query_Prep',
        description=(
            'Tool to generate a MongoDB query from user input. Ensure user input is complete and accurate before invoking this tool.'
        )
    )

    # Define the toolset (add more tools here if needed)
    tools = [prepare_input_tool, read_mongodb_tool]

    # Initialize the OpenAI language model
    llm = OpenAI(model='gpt-3.5-turbo')

    # System prompt for the agent
    SYSTEM_PROMPT = """
    You are a diligent flight assistant. Your job is to assist users in retrieving accurate and up-to-date 
    flight information from the database. Follow these guidelines:
    
    1. **Clarify User Input**:
       - If the user's input is unclear or lacks sufficient details to form a query, ask follow-up questions.
       - Confirm with the user before proceeding if you are unsure.

    2. **Prepare Database Queries**:
       - Use the `Query_Prep` tool to generate a proper MongoDB query from the user's input. Result in json format.

    3. **Retrieve Information**:
       - Use the `MongoDB_Retriever` tool to fetch real-time data from the database.
       - Do not rely on assumptions or generate fake data; only provide verified information.

    4. **Communicate Clearly**:
       - Respond to the user with information retrieved from the database in a concise and professional manner.
       - Offer additional assistance if necessary.
    """.strip()

    # Initialize the agent with tools and prompt
    agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True, system_prompt=SYSTEM_PROMPT)

    def cleanup():
        pass

    try:
        print("Flight assistant is ready to assist. Type your query below:")
        while True:
            user_input = input("USER: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye! Have a great day!")
                cleanup()
                break

            # Run the agent and handle user queries
            response = agent.chat(user_input)
            print(f"AGENT: {response}")
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
