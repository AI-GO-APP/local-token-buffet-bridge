import json
import time
import uuid
import pytest
from desktop_bridge.store import Store
from desktop_bridge.service import Runtime
from desktop_bridge.errors import BridgeError
from desktop_bridge.contract import validate

BODY={'model':'openai/gpt-6-astra','messages':[{'role':'user','content':'synthetic'}]}
TOOLS=[{'type':'function','function':{'name':'sum','parameters':{'type':'object','properties':{'a':{'type':'integer'}},'required':['a'],'additionalProperties':False}}}]

@pytest.mark.parametrize('option',[None,{}, {'data_collection':'deny'},{'data_collection':'allow'}])
def test_provider_metadata_does_not_gate_inference(option,tmp_path):
    store=Store(tmp_path);_,key=store.create_key('cloud');calls=[]
    def runner(*args):calls.append(1);return 'ok',{'prompt_tokens':10,'completion_tokens':1},'stop'
    rt=Runtime(store,runner)
    try:
        request={**BODY,'provider':option}
        rt.preflight(request,store.authenticate(key))
        task=rt.submit(request,store.authenticate(key),20)
        assert task.done.wait(5) and not task.error and len(calls)==1
    finally:rt.stop();store.close()

@pytest.mark.parametrize('override',[
    {'tools':[{'type':'function','function':{'name':'sum','parameters':True}}]},
    {'tool_choice':{'type':'function','function':{'name':[]}}},
    {'messages':[{'role':'tool','tool_call_id':[],'content':'4'}]},
    {'messages':[{'role':'assistant','tool_calls':42}]},
])
def test_malformed_tools_return_contract_error(override,tmp_path):
    store=Store(tmp_path);_,key=store.create_key('contract')
    try:
        with pytest.raises(BridgeError) as error:validate({**BODY,'tools':TOOLS,**override},store,store.authenticate(key))
        assert error.value.status==400
    finally:store.close()

def test_idempotent_call_and_conflict_survive_restart(tmp_path):
    store=Store(tmp_path);_,key=store.create_key('cloud');caller=store.authenticate(key);calls=[]
    def runner(*args):calls.append(1);return 'ok',{'prompt_tokens':10,'completion_tokens':1},'stop'
    rt=Runtime(store,runner);rid=str(uuid.uuid4())
    task=rt.submit(BODY,caller,20,rid);assert task.done.wait(5)
    assert rt.submit(BODY,caller,20,rid) is task and len(calls)==1
    with pytest.raises(BridgeError) as conflict:rt.submit({**BODY,'messages':[{'role':'user','content':'different'}]},caller,20,rid)
    assert conflict.value.status==409
    rt.stop();store.close();store=Store(tmp_path);rt=Runtime(store,runner)
    with pytest.raises(BridgeError) as prior:rt.submit(BODY,store.authenticate(key),20,rid)
    assert prior.value.status==409 and len(calls)==1
    rt.stop();store.close()

def test_preflight_validates_without_inference_or_billing(tmp_path):
    s=Store(tmp_path);_,key=s.create_key('cloud');rt=Runtime(s,lambda *_:pytest.fail('inference'))
    assert rt.preflight(BODY,s.authenticate(key)).model['id']==BODY['model']
    assert s.db.execute('select count(*) from requests').fetchone()[0]==0
    s.set('accepting',False)
    with pytest.raises(BridgeError) as err:rt.preflight(BODY,s.authenticate(key))
    assert err.value.status==503;rt.stop();s.close()


def test_receipt_recovers_ram_result_and_durable_billing_without_reinference(tmp_path):
    s=Store(tmp_path);_,key=s.create_key('cloud');_,other=s.create_key('other');calls=[]
    def runner(*_):calls.append(1);return 'ok',{'prompt_tokens':10,'completion_tokens':1},'stop'
    rt=Runtime(s,runner);caller=s.authenticate(key);rid=str(uuid.uuid4())
    t=rt.submit(BODY,caller,20,rid);assert t.done.wait(5)
    receipt=rt.receipt(rid,caller)
    assert receipt['result']==t.result and receipt['billing']['usage']['cost']==t.result['usage']['cost']
    with pytest.raises(BridgeError):rt.receipt(rid,s.authenticate(other))
    rt.tasks.clear()
    expired=rt.receipt(rid,caller)
    assert 'result' not in expired and expired['billing']==receipt['billing'] and len(calls)==1
    assert all(name not in dict(s.db.execute('select * from requests').fetchone()) for name in ('messages','response','content'))
    rt.stop();s.close()

def test_deadline_expired_before_execution(tmp_path):
    s=Store(tmp_path);_,key=s.create_key('cloud');rt=Runtime(s,lambda *_:pytest.fail('inference'))
    t=rt.submit(BODY,s.authenticate(key),20,deadline=time.time()-1);assert t.done.wait(3)
    assert t.error.status==408;rt.stop();s.close()

def test_tool_proposals_and_history_are_validated_and_never_executed(tmp_path):
    s=Store(tmp_path);_,key=s.create_key('cloud');seen=[]
    def runner(spec,*_):seen.append(spec);return json.dumps({'content':None,'calls':[{'name':'sum','arguments':'{"a":4}'}]}),{'prompt_tokens':10,'completion_tokens':1},'stop'
    rt=Runtime(s,runner)
    t=rt.submit({**BODY,'tools':TOOLS,'tool_choice':'required','parallel_tool_calls':False},s.authenticate(key),30)
    assert t.done.wait(5) and not t.error
    msg=t.result['choices'][0]['message'];assert t.result['choices'][0]['finish_reason']=='tool_calls'
    call=msg['tool_calls'][0];assert json.loads(call['function']['arguments'])=={'a':4}
    follow={**BODY,'tools':TOOLS,'messages':BODY['messages']+[msg,{'role':'tool','tool_call_id':call['id'],'content':'4'}]}
    assert validate(follow,s,s.authenticate(key)).tool_proposals
    # Enforced output schema prevents an unregistered function from escaping.
    rt.runner=lambda *_:(json.dumps({'content':None,'calls':[{'name':'shell','arguments':'{}'}]}),{'prompt_tokens':1,'completion_tokens':1},'stop')
    bad=rt.submit({**BODY,'tools':TOOLS},s.authenticate(key),20);assert bad.done.wait(5) and bad.error.status==502
    rt.stop();s.close()


def test_full_agent_tool_count_and_invalid_arguments_keep_real_usage(tmp_path):
    s=Store(tmp_path);_,key=s.create_key('agent');caller=s.authenticate(key)
    tools=[{'type':'function','function':{'name':'tool_'+str(n),'parameters':TOOLS[0]['function']['parameters']}} for n in range(131)]
    spec=validate({**BODY,'tools':tools},s,caller)
    assert len(spec.tool_schemas)==131 and len(json.dumps(spec.schema))<5000
    def runner(*_):return json.dumps({'content':None,'calls':[{'name':'tool_0','arguments':'{"a":"bad"}'}]}),{'prompt_tokens':10,'completion_tokens':1},'stop'
    rt=Runtime(s,runner);rid=str(uuid.uuid4());t=rt.submit({**BODY,'tools':tools},caller,30,rid)
    assert t.done.wait(5) and t.error.status==502
    receipt=rt.receipt(rid,caller)
    assert 'result' not in receipt and receipt['billing']['usage']['cost']>0
    assert receipt['billing']['usage']['prompt_tokens']==10
    rt.stop();s.close()


# Synthetic 1x1 PNG; tool mode used to reject any image ("工具模式暫僅接受文字").
PNG='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg=='
IMG={'type':'image_url','image_url':{'url':PNG}}

def test_tool_mode_attaches_user_images_after_serialized_conversation(tmp_path):
    s=Store(tmp_path);_,key=s.create_key('agent')
    try:
        msgs=[{'role':'user','content':[{'type':'text','text':'synthetic screen'},IMG]},
              {'role':'assistant','content':None,'tool_calls':[{'id':'c1','type':'function','function':{'name':'sum','arguments':'{"a":1}'}}]},
              {'role':'tool','tool_call_id':'c1','content':'1'},
              {'role':'user','content':[IMG,{'type':'text','text':'compare'}]}]
        spec=validate({**BODY,'tools':TOOLS,'messages':msgs},s,s.authenticate(key))
        parts=spec.messages[-1]['parts']
        assert [p['type'] for p in parts]==['text','image_url','image_url']
        assert all(p['image_url']['url']==PNG for p in parts[1:])
        serialized=json.loads(parts[0]['text'])['messages']
        assert serialized[0]['content']=='synthetic screen\n[image 1]' and serialized[3]['content']=='[image 2]\ncompare'
        assert PNG not in parts[0]['text'] and spec.tool_proposals
    finally:s.close()

@pytest.mark.parametrize('messages,status',[
    ([{'role':'system','content':[IMG]},{'role':'user','content':'x'}],400),
    ([{'role':'user','content':[{'type':'image_url','image_url':{'url':'https://example.invalid/a.png'}}]}],400),
    ([{'role':'user','content':[{'type':'audio','audio':{}}]}],400),
    ([{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'}}]}],400),
])
def test_tool_mode_image_rules(messages,status,tmp_path):
    s=Store(tmp_path);_,key=s.create_key('agent')
    try:
        with pytest.raises(BridgeError) as error:validate({**BODY,'tools':TOOLS,'messages':messages},s,s.authenticate(key))
        assert error.value.status==status
    finally:s.close()


def _image(decoded_bytes):
    import base64
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(bytes([137])*decoded_bytes).decode()}}

def test_tool_mode_has_no_bridge_image_count(tmp_path):
    # D-50: the former 8-image cap was Bridge policy; Chat mode never had one.
    s=Store(tmp_path);_,key=s.create_key('agent')
    try:
        spec=validate({**BODY,'tools':TOOLS,'messages':[{'role':'user','content':[IMG]*20}]},s,s.authenticate(key))
        assert [p['type'] for p in spec.messages[-1]['parts']].count('image_url')==20
    finally:s.close()

def test_image_ceiling_follows_provider_upstream(tmp_path):
    from desktop_bridge.limits import CLAUDE_MAX_IMAGE_BASE64
    s=Store(tmp_path);_,key=s.create_key('agent');caller=s.authenticate(key)
    try:
        # Above the former 2 MB Bridge cap: Codex resizes images itself; no per-image Bridge cap.
        big=_image(6_000_000)
        assert validate({**BODY,'messages':[{'role':'user','content':[big]}]},s,caller).messages[-1]['parts'][0]['type']=='image_url'
        claude={**BODY,'model':'anthropic/claude-haiku-4.5'}
        largest=CLAUDE_MAX_IMAGE_BASE64//4*3  # base64 length exactly at the Claude API ceiling
        validate({**claude,'messages':[{'role':'user','content':[_image(largest)]}]},s,caller)
        with pytest.raises(BridgeError) as error:
            validate({**claude,'messages':[{'role':'user','content':[_image(largest+3)]}]},s,caller)
        assert error.value.status==413
    finally:s.close()
