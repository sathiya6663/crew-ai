from crewai.project import CrewBase, agent, task, crew
from crewai import Agent, Task, Crew, Process, LLM
import os
os.environ["CREWAI_DISABLE_IPYTHON"] = "true"

from dotenv import load_dotenv
 
load_dotenv()

GEMINI_MODEL = "gemini/gemini-3.1-flash-lite"
# Initialize Gemini 2.5 Flash model using CrewAI's LLM class
gemini_llm = LLM(
    model=GEMINI_MODEL,
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)
@CrewBase
class AiTrendCrew:
    """AiTrendCrew system orchestration using YAML files"""
    
    # Path paths relative to where this class module sits
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            allow_delegation=False,
            llm=gemini_llm
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'],
            verbose=True,
            allow_delegation=False,
            llm=gemini_llm
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['writing_task']
            # Context mapping is handled automatically by the order listed in the crew 
            # or you can add context=[self.research_task()] explicitly if needed.
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,       # Automatically collected from @agent decorators
            tasks=self.tasks,         # Automatically collected from @task decorators
            process=Process.sequential,  
            verbose=True
        )  
        
# if __name__ == "__main__":
#     print("Starting AI Trend Analysis Crew...")
    
#     # Instantiate the decorated class and trigger the crew method
#     ai_crew_instance = AiTrendCrew().crew()
#     result = ai_crew_instance.kickoff()
    
#     print("\n" + "="*50)
#     print("FINAL RESULT:")
#     print("="*50)
#     print(result)    
# 
from fastapi import FastAPI, HTTPException

app = FastAPI(title="CrewAI Azure Service")

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "CrewAI Runner"}

@app.post("/run-crew")
def run_crew():
    try:
        result = AiTrendCrew().crew().kickoff()
        return {"status": "success", "result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
        
        
        
        
        