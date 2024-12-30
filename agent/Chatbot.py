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
from pydantic import BaseModel
from enum import Enum
import json
from bson import ObjectId

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
    #departure_time:Optional[str] = None
    flight_id:Optional[str] = None
    counter:Optional[str] = None
    gate:Optional[str] = None
    departure_region_name:Optional[str] = None
    arrival_region_name:Optional[str] = None
    departure_airport:Optional[str] = None
    arrival_airport:Optional[str] = None
    airline:Optional[str] = None
'''
schema = StructType([
        StructField("departure_time", StringType(), True),
        StructField("flight_id", StringType(), True),
        StructField("counter", StringType(), True),
        StructField("gate", StringType(), True),
        StructField("departure_region_name", StringType(), True),
        StructField("arrival_region_name", StringType(), True),
        StructField("departure_airport", StringType(), True),
        StructField("arrival_airport", StringType(), True),
        StructField("airline", StringType(), True)
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


class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)


class MongoDBflow(Workflow):
    @step 
    async def start(self, ctx: Context ,ev: StartEvent) -> ConnectDB_Event | QueryGeneration_Event :

        MONGO_DB = "flight_db"  
        COLLECTION_NAME = "flight_info"  
        CONNECTION_STRING = "mongodb://localhost:27017" 

        LLM = OpenAI(
            model='gpt-3.5-turbo',
            logprobs=None,  
            default_headers={},  
        )
    
        await ctx.set('MONGO_DB', MONGO_DB)
        await ctx.set('COLLECTION_NAME', COLLECTION_NAME)
        await ctx.set('CONNECTION_STRING', CONNECTION_STRING)
        await ctx.set('LLM', LLM)

        try:
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

        if client:
            client.close()
            print("MongoDB client closed.")
        
        if not retrieved_data: # empty list
            formatted_string = 'No data retrieved, tell user there is no flights available.'
        else: 
            formatted_string = "\n".join(json.dumps(item, indent=4, cls=MongoEncoder) for item in retrieved_data)
        
        #print(formatted_string)
        observation = f"Retrieved data: \n{formatted_string}"
        return StopEvent(result=observation)

#draw_all_possible_flows(MongoDBflow)
# - `route` (string): The flight's route, in the format **"from <destination1> to <destination2>"**. Both `destination1` and `destination2` must be provided. For example, "from NYC to LAX".
# "route": "from NYC to LAX",


class IntentType(str, Enum):
    QUERY = "query"
    BOOKING = "booking"
    REGULATION = "regulation"
    UNKNOWN = "unknown"

class IntentClassification(BaseModel):
    intent: IntentType
    confidence: float = 1.0

intent_classification_prompt = PromptTemplate(
    """
    As a flight assistant, classify the user's intent into one of these categories:
    1. QUERY - User wants to search for flight information (schedules, status, etc.)
    2. BOOKING - User wants to book a flight
    3. REGULATION - User is asking about airline regulations or travel requirements
    4. UNKNOWN - Cannot determine the user's intent clearly

    User input: {text}

    Respond with the intent classification in the following JSON format:
    {
        "intent": "QUERY|BOOKING|REGULATION|UNKNOWN",
        "confidence": <float between 0 and 1>
    }
    """
)

class GatherInformation(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> ParseInput_Event | StopEvent:
        LLM = OpenAI(
            model='gpt-3.5-turbo',
            logprobs=None,
            default_headers={},
        )
        await ctx.set('LLM', LLM)
        print(f'User input: {ev.query}')
        
        intent_classification = LLM.structured_predict(
            IntentClassification,
            intent_classification_prompt,
            text=ev.query
        )
        
        print(f"Detected intent: {intent_classification.intent} with confidence: {intent_classification.confidence}")
        
        if intent_classification.intent == IntentType.QUERY:
            await ctx.set('user_query', ev.query)
            return ParseInput_Event(payload=ev.query)
        elif intent_classification.intent == IntentType.BOOKING:
            return StopEvent(result="I apologize, but flight booking is not available at the moment. Please contact our customer service or visit our website for booking assistance.")
        elif intent_classification.intent == IntentType.REGULATION:
            regulation_prompt = PromptTemplate(
                """
                As a flight assistant, provide information about the airline regulation or requirement that the user is asking about.
                Be concise but comprehensive in your response.
                
                User query: {text}
                
                Provide a helpful response about the relevant regulations:
                """
            )
            regulation_response = LLM.complete(regulation_prompt, text=ev.query)
            return StopEvent(result=str(regulation_response))
        else:
            return StopEvent(result="I'm not sure what you're asking for. Could you please rephrase your question? You can ask me about flight schedules, airline regulations, or general flight information.")

    @step
    async def parse_userQuery(self, ctx: Context, ev: ParseInput_Event) -> QueryGeneration_Event:

        simplified_prompt = PromptTemplate(
    """
    You are a helpful flight attendant. If you encounter time-related terms like 'yesterday', 'tomorrow', 'today', or a specific date, extract them as they are and map them to `departure_time` or `arrival_time` where applicable.
    Extract the following information from the query below. If any field is not explicitly mentioned in the query, set it as `null`. Follow the schema strictly for the JSON response:
    
    **Schema:**
    - `date` (string): The date associated with the flight, including terms like 'today', 'tomorrow', or specific dates (e.g., '2024-12-25').
    - `scheduled_time` (string): The scheduled time of the flight, if mentioned (e.g., '14:30').
    - `flight_id` (string): The flight's identifier.
    - `status` (string): The flight's status. (always query flights have status open)
    - `departure_region_name` (string): the departure region.(non-accented form)
    - `arrival_region_name` (string): the arrival region.(non-accented form)
    - `airline` (string): the flight's airline.(standardized international airline name)
    Example Query:
    "i want to find flights from Đà Nẵng to Hồng Kông tomorrow by vietjet."

    Example Response:
    ```json
    {
        "date": "tomorrow",
        "scheduled_time": null,
        "flight_id": null,
        "counter": null,
        "gate": null,
        "status": "OPN",
        "departure_region_name": "Da Nang",
        "arrival_region_name": "Hong Kong",
        "airline": "VietJet Air"
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
        
        departTime = tmp.get('departure_time', None)
        scheduleTime = tmp.get('scheduled_time', None)  # Default to None if the key doesn't exist
        updatedTime = tmp.get('updated_time', None)
        dateTime = tmp.get('date', None)

        if departTime is not None:
            parsedDepart = parseTime(departTime)
            tmp['departure_time'] = parsedDepart
        
        if dateTime is not None:
            parsedDateTime = parseTime(dateTime)
            tmp['date'] = parsedDateTime

        if scheduleTime is not None:
            parsedSchedule = parseTime(scheduleTime)
            tmp['scheduled_time'] = parsedSchedule
            
        if updatedTime is not None:
            parsedUpdated = parseTime(updatedTime)
            tmp['updated_time'] = parsedUpdated

        mongoDB_query = tmp
        
        await ctx.set('mongoDB_query', mongoDB_query)
        print(f"Final MongoDB query: {mongoDB_query}")

        return QueryGenerationComplete_Event(payload='')


    @step
    async def query_output(self, ctx: Context, ev: QueryGenerationComplete_Event) -> StopEvent:
        mongoDB_query = await ctx.get('mongoDB_query')
        observation = f'MongDB query: {mongoDB_query}'
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
    llm = OpenAI(
        model='gpt-3.5-turbo',
        logprobs=None,  # Thêm tham số này
        default_headers={},  # Thêm tham số này
    )

    # System prompt for the agent
    SYSTEM_PROMPT = """
    You are a diligent flight assistant. Your job is to assist users in retrieving accurate and up-to-date 
    flight information from the database. Follow these guidelines:
    
    1. **Clarify User Input**:
       - User must provide information about departure date, departure region, arrival region before proceed to the retrieving step.
       - If the user's input is unclear or lacks sufficient details to form a query, ask follow-up questions then from a proper final user input string.
       - Confirm with the user before proceeding if you are unsure.
       - Final user input must be in natural language, do not perform any conversion, keep everything as it is.

    2. **Prepare Database Queries**:
       - Use the `Query_Prep` tool to generate a proper MongoDB query from the user's input. Result in json format.

    3. **Retrieve Information**:
       - Use the `MongoDB_Retriever` tool to fetch real-time data from the database.
       - Do not rely on assumptions or generate fake data; only provide verified information.

    4. **Communicate Clearly**:
       - Respond to the user with information retrieved from the database in a concise and professional manner.
       - Offer additional assistance if necessary.
    """.strip()
    
    DEMO_PROPMPT = '''
    You are a diligent and professional flight assistant tasked with retrieving accurate and up-to-date flight information. Follow these step-by-step guidelines to assist users effectively:
    1. **Clarify User Input**:
        - Ensure the user provides the following mandatory information, if not, ask for confirmation on mandatory details:
            - **Departure date**: The user should mention when they want to travel (e.g., "today," "tomorrow," or a specific date), must provide one date-related information.
            - **Departure region**: The user should specify where they are departing from (e.g., "Ha Noi"), must provide one specific region.
            - **Arrival region**: The user should specify the destination (e.g., "Da Nang"), must provide one specific region.
        - If all mandatory details (Departure date, Departure region, Arrival region) are provided, with included any optional details (such as the airline). Then, immediatly move to step 2 with the most updated information, explicitly listing every detail in the final output, without reconfirming with user.
        - If you are unsure about anything, then ask user for confirmation:
            - Example: "Thank you! You are searching for flights from DAD to HAN on tomorrow by Vietnam Airlines. Is that correct?"
            - Ensure that the **final user input remains in its original, natural language form** (e.g., "flights from DAD to HAN tomorrow by Vietnam Airlines").
        - If any required details are missing:
            - Politely ask follow-up questions while retaining and using previously provided information, **ensure the final query includes all previously stated and newly provided details**:.
                - If mandatory details were already given:
                    - Example 1: The user previously mentioned "from Da Nang to Ha Noi" and now adds "today." Combine them: "You are searching for flights from Da Nang to Ha Noi today. Is that correct?"       
        - If the user provides terms like "tomorrow," "today," "next week," etc., you **do not convert** these terms to a date. Keep them as is.
        
    2. **Prepare Database Queries**:
        - Once the user has provided a complete query with all necessary information, pass the **exact same query** (in natural language format) to the **Query_Prep** tool. The format should be as stated by the user (e.g., "flights from DAD to HAN tomorrow").
        - Use the `Query_Prep` tool to create a MongoDB query that matches the user's input. 
        - The query must:
            - Be in JSON format.
            - Include all mandatory fields (departure date, departure region, arrival region).
        - Example of a properly formatted query:
        ```json
        {
            "departure_region": "JFK",
            "arrival_region": "LAX",
            "departure_date": "2024-12-25"
        }
        ```

    3. **Retrieve Flight Information**:
        - Use the `MongoDB_Retriever` tool to fetch data based on the prepared query.
        - Ensure all information retrieved is accurate and verified.
        - Do not generate or assume flight data.

    4. **Respond to the User**:
        - Clearly communicate the retrieved information. Include:
            - Flight number
            - Departure and arrival times
            - Departure and arrival airports
            - Gate and terminal details (if available)
            - Flight status (e.g., on-time, delayed)
        - Example response:
            - "Here is the flight information for your query: Flight DL123 will depart from JFK on 2024-12-25 at 14:00 and arrive at LAX. Terminal 4, Gate 23. Status: On-Time."
        - Offer further assistance if necessary:
            - "Would you like me to help with another flight search?"

    5. **Handle Errors Gracefully**:
        - If no flights are found, respond politely:
            - Example: "I couldn’t find any flights matching your criteria. Would you like to adjust the search?"
        - If the database retrieval fails, apologize and suggest trying again:
            - Example: "I'm having trouble retrieving the flight data right now. Can I try again for you?"

    6. **Follow Professional Communication Standards**:
        - Be concise and polite in all responses.
        - Avoid technical jargon when speaking to users.
        - Always prioritize the user's needs and provide additional help where possible.
    '''.strip()

    # Initialize the agent with tools and prompt
    agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True, system_prompt=DEMO_PROPMPT)

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
