from src.llmconfig import models
from langchain_core.messages import AIMessage
from ..prompt import *
from ..state import *
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel,Field
from langgraph.types import Command
from src.llm_wrapper import llm_wrapper

#大模型选择
ds_V3 = models["DeepSeek_V3"]
ds_R1 = models["DeepSeek_R1"]
kimi_k2_5 = models["KIMI_K2.5"]

class focus_response_format(BaseModel):
    focus: list[str] = Field(description= "归纳的争议焦点列表")
    messages: str = Field(description= "法官对于争议焦点归纳的说明")


async def judge_open(state: CourtState) -> CourtState:
    """
    【静态节点】审判长宣布开庭
    需要消费状态信息：location,defendant_name
    """
    judge_announcement= await JUDGE_ANNOUNCEMENT.ainvoke(
        {
            "court_name" : state.meta.court_name,
            "defendant_name" : state.meta.defendant_name
        }
    )
    # Add name field to each message
    messages = judge_announcement.to_messages()
    for msg in messages:
        msg.name = f"审判长{state.meta.judge_name}"
    return {
        "messages" : messages
    }
    
async def judge_check(state: CourtState) -> CourtState:
    """
    【动态AI节点】审判长核对被告人信息，
    需要消费状态中关于被告的一系列信息，调用LLM产生问句
    """
    judge_check = await JUDGE_CHECK.ainvoke(
        {
            "name": state.meta.defendant_name,
            "former_name": state.meta.defendant_former_name,
            "birthdate": state.meta.defendant_birthdate,
            "birthplace": state.meta.defendant_birthplace,
            "ethnicity": state.meta.defendant_ethnicity,
            "education": state.meta.defendant_education,
            "occupation": state.meta.defendant_occupation,
            "employer": state.meta.defendant_employer,
            "residence": state.meta.defendant_residence,
            "ID_number": state.meta.defendant_ID_number,
            "legal_record": state.meta.defendant_legal_record,
            "detention_date": state.meta.detention_date,
            "indictment_date": state.meta.indictment_date
        }
    )
    judge = create_agent(model=ds_V3)
    output = await llm_wrapper.ainvoke_with_retry(
        judge.ainvoke,
        judge_check
    )
    question = output.get("messages", [])[-1]
    question.name = f"审判长_{state.meta.judge_name}"
    return {
        "messages" : [question,
                      AIMessage(content="审判长，以上情况属实。",
                                  name=f"被告人{state.meta.defendant_name}"
                                )
                        ]
    }

async def right_notify(state: CourtState) -> CourtState:
    """
    【静态节点】审判长介绍开庭人员，并告知被告人基本诉讼权利
    """
    introduction = await INTRODUCE.ainvoke(
        {
            "court_name": state.meta.court_name,
            "prosecutor_title": state.meta.prosecutor_title,
            "defendant_name": state.meta.prosecutor_name,
            "crime": state.meta.crime,
            "judge_name": state.meta.judge_name,
            "juede_name2": state.meta.judge_name_2,
            "clerk_name": state.meta.clerk_name,
            "prosecutor_name": state.meta.prosecutor_name
        }
    )
    # Add name field to each message from introduction
    intro_messages = introduction.to_messages()
    for msg in intro_messages:
        msg.name = f"审判长{state.meta.judge_name}"
    notice = AIMessage(content=RIGHT_NOTIFY,
                         name = f"审判长{state.meta.judge_name}")
    return {
        "messages": intro_messages + [notice]
    }

async def judge_start_evidence(state: CourtState) -> CourtState:
    """
    【静态节点】审判长宣布开始进行交叉质证环节
    """
    return {
        "messages" : AIMessage(content=JUDGE_START_EVIDENCE,name=f"审判长{state.meta.judge_name}")
    }

async def judge_confirm(state: CourtState)-> CourtState:
    """
    【静态节点】审判长询问辩护人是否有补充证据意见
    """    
    return {
        "messages" : AIMessage(content= JUDGE_CONFIRM, name=f"审判长{state.meta.judge_name}")
    }

async def judge_start_debate(state:CourtState)-> CourtState:
    """
    【静态节点】审判长宣布法庭调查结束，进入法庭辩论环节
    """
    return {
        "messages": AIMessage(content= JUDGE_START_DEBATE,name= f"审判长{state.meta.judge_name}"),
        "phase": PhaseEnum.DEBATE
    }

async def judge_summary(state: CourtState)-> CourtState:
    """
    【动态AI节点】审判长根据三方发言，归纳争议焦点，并作说明
    """
    judge = create_agent(model= kimi_k2_5,response_format= ToolStrategy(focus_response_format))
    judge_summary = await JUDGE_SUMMARY.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    response = await llm_wrapper.ainvoke_with_retry(
        judge.ainvoke,
        judge_summary
    )
    summary: focus_response_format = response["structured_response"]
    # Wrap the summary message in AIMessage with name field
    summary_msg = AIMessage(content=summary.messages, name=f"审判长{state.meta.judge_name}")
    return {
        "focus": summary.focus,
        "messages": [summary_msg],
        "phase": PhaseEnum.VERDICT
    }

async def focus(state: CourtState)-> Command:
    """
    【路由节点】审判长根据争议焦点的索引切换争议焦点
    """
    try:
        currrent_focus = state.focus[state.focus_index]
    except IndexError:
        currrent_focus = None

    if currrent_focus:
        return Command(
            update={
                "messages": [AIMessage(content= f"合议庭已经充分了解双方意见，现在进行争议焦点``{state.focus[state.focus_index]}``的解决。",
                                       name= f"审判长{state.meta.judge_name}")],
                "pros_focus_rounds": F_ROUNDS
            },
            goto= "pros_focus"
        )
    else:
        return Command(
            update={
                "messages": [AIMessage(content= "法庭辩论结束，现在由公诉人作最后的总结。",
                                       name = f"审判长{state.meta.judge_name}")]
            },
            goto= "pros_sumup"
        )

async def judge_verdict(state:CourtState)-> CourtState:
    """
    【动态AI节点】审判长根据整个开庭过程做出裁决。
    """
    judge = create_agent(model=ds_V3)
    judge_verdict = await JUDGE_VERDICT.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    response = await llm_wrapper.ainvoke_with_retry(
        judge.ainvoke,
        judge_verdict
    )
    verdict = response.get("messages",[])[-1]
    # Add name field to the verdict message
    verdict.name = f"审判长{state.meta.judge_name}"
    return {
        "messages": [verdict]
    }