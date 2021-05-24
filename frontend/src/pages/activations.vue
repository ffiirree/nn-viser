<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" @change="update">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="update">
                    <el-option value='static/images/cat_dog.png'/>
                    <el-option value='static/images/spider.png'/>
                    <el-option value='static/images/snake.jpg'/>
                </el-select>
            </div>
        </div>
        <div class="network">
            <div class="unit">
                <div class="layer">
                    <div class="name">input</div>
                    <img :src="params.input" crossorigin='anonymous'/>
                </div>
            </div>
            <div class="unit" v-for="(unit, index) in res.units" :key="index">
                <div class="layer" v-for="(layer, name) in unit.layers" :key="name">
                    <div class="name">{{name}}</div>
                    <div class="channels" v-for="(value, name) in layer.channels" :key="name">
                        <img :src="value.path" crossorigin='anonymous'/>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            models: [],
            res: [],
            params: {
                model: 'alexnet',
                input: 'static/images/cat_dog.png',
            }
        };
    },
    created() {
        this.config()
        this.update()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        response_activations(data) {
            console.log(data)
            this.res = data
            this.$forceUpdate()
        }
    },
    methods: {
        config() {
            this.$socket.emit("get_models")
        },
        update() {
            this.$socket.emit("activations", this.params);
        },
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        display: flex;

        .unit {
            display: flex;
            padding: 0 5px;

            .layer {
                width: 70px;
                display: flex;
                flex-flow: column;
                align-items: center;

                img {
                    width: 64px;
                    height: 64px;
                }
            }
        }
    }
}

</style>
